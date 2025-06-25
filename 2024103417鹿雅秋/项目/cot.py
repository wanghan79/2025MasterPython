import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline


class GenerationTracker:
    """跟踪生成状态和模板使用情况"""

    def __init__(self, save_dir="generation_logs"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.stats = {
            'llm_generations': 0,
            'template_generations': 0,
            'failed_generations': 0,
            'generation_times': [],
            'prompts': [],
            'responses': []
        }

    def log_generation(self, generation_type, prompt, response, time_taken, error=None):
        """记录单次生成事件"""
        if generation_type == 'llm':
            self.stats['llm_generations'] += 1
        elif generation_type == 'template':
            self.stats['template_generations'] += 1

        if error:
            self.stats['failed_generations'] += 1

        self.stats['generation_times'].append(time_taken)
        self.stats['prompts'].append(prompt)
        self.stats['responses'].append(response)

    def get_summary(self):
        """获取生成统计摘要"""
        total = self.stats['llm_generations'] + self.stats['template_generations']
        if total == 0:
            return "No generations recorded"
        return {
            'Total Generations': total,
            'LLM Generations': self.stats['llm_generations'],
            'Template Generations': self.stats['template_generations'],
            'Failed Generations': self.stats['failed_generations'],
            'Average Generation Time': f"{np.mean(self.stats['generation_times']):.3f}s"
        }


class CoTTrajectoryPredictor(nn.Module):
    """基于思维链的轨迹预测器"""

    def __init__(self, device=None, hidden_size=64):
        super().__init__()
        # 设置设备
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 使用本地的phi-2模型
        self.llm_model_name = "./phi2"  # 修改为本地路径

        # 初始化生成跟踪器
        self.generation_tracker = GenerationTracker()

        print("正在加载语言模型...")
        try:
            print(f"尝试从路径加载模型: {self.llm_model_name}")

            # 检查模型路径是否存在
            if not os.path.exists(self.llm_model_name):
                raise FileNotFoundError(f"模型路径不存在: {self.llm_model_name}")

            # 检查必要的模型文件
            required_files = ['config.json', 'tokenizer.json']
            model_files = [
                ['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors'],  # 分片模型文件
                ['model.safetensors'],  # 单个模型文件
                ['pytorch_model.bin']  # PyTorch格式
            ]

            # 检查基本配置文件
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(self.llm_model_name, f))]
            if missing_files:
                raise FileNotFoundError(f"缺少必要的配置文件: {', '.join(missing_files)}")

            # 检查模型文件
            model_files_exist = False
            for file_group in model_files:
                if all(os.path.exists(os.path.join(self.llm_model_name, f)) for f in file_group):
                    model_files_exist = True
                    print(f"找到模型文件: {', '.join(file_group)}")
                    break

            if not model_files_exist:
                raise FileNotFoundError(f"未找到完整的模型文件，需要以下任一组合之一:\n" +
                                        "\n".join([f"- {', '.join(group)}" for group in model_files]))

            print("加载tokenizer...")
            # 加载本地模型和分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.llm_model_name,
                trust_remote_code=True,
                local_files_only=True,  # 强制使用本地文件
                use_fast=False  # 使用慢速但更稳定的tokenizer
            )
            
            # 设置padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("已设置pad_token为eos_token")
            
            print("Tokenizer加载成功")

            print("加载模型...")
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                trust_remote_code=True,
                local_files_only=True,  # 强制使用本地文件
                torch_dtype=torch.float32,  # 明确指定数据类型
                use_safetensors=True  # 启用safetensors支持
            )
            print("模型加载成功")

            # 将模型移动到指定设备
            print(f"将模型移动到设备: {self.device}")
            self.llm = self.llm.to(self.device)

            # 打印模型结构
            print("\n模型结构:")
            for name, _ in self.llm.named_modules():
                print(name)

            print("\n配置LoRA...")
            # 配置LoRA
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["mlp.dense_4h_to_h", "mlp.dense_h_to_4h", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.dense"],  # 更新的phi-2模块名称
                bias="none",
                layers_to_transform=None  # 应用到所有层
            )

            # 应用LoRA
            print("应用LoRA配置...")
            self.llm = get_peft_model(self.llm, peft_config)

            # 冻结基础模型参数，解冻LoRA参数
            print("设置参数梯度...")
            for name, param in self.llm.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            # 创建优化器
            print("创建优化器...")
            self.optimizer = torch.optim.AdamW(
                [p for p in self.llm.parameters() if p.requires_grad],
                lr=1e-4,
                weight_decay=0.01
            )

            # 创建pipeline
            print("创建生成pipeline...")
            self.pipe = pipeline(
                "text-generation",
                model=self.llm,
                tokenizer=self.tokenizer,
                device_map={"": self.device},  # 确保所有组件都在同一设备上
                torch_dtype=torch.float32
            )

            # 打印LoRA统计信息
            trainable_params = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.llm.parameters())
            print(f"LoRA统计信息:")
            print(f"可训练参数数量: {trainable_params:,d}")
            print(f"总参数数量: {total_params:,d}")
            print(f"可训练参数比例: {100 * trainable_params / total_params:.2f}%")
            print("模型初始化完成")

        except FileNotFoundError as e:
            print(f"文件未找到错误: {str(e)}")
            raise
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            print(f"错误类型: {type(e).__name__}")
            raise

        # 特征归一化层
        self.feature_norm = nn.BatchNorm1d(4)

        # 增强的轨迹编码器，使用LSTM
        self.trajectory_encoder = nn.ModuleDict({
            'spatial': nn.Sequential(
                nn.Linear(4, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ),
            'temporal': nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=2,
                batch_first=True,
                dropout=0.2,
                bidirectional=True
            ),
            'attention': nn.MultiheadAttention(
                embed_dim=hidden_size * 2,
                num_heads=4,
                dropout=0.1
            ),
            'fusion': nn.Sequential(
                nn.Linear(hidden_size * 4, hidden_size * 2),
                nn.LayerNorm(hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        })

        # 多尺度轨迹预测器
        self.trajectory_predictors = nn.ModuleDict({
            'short_term': nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, 16)  # 4个特征 * 4个时间步
            ),
            'mid_term': nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, 24)  # 4个特征 * 6个时间步
            ),
            'long_term': nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, 40)  # 4个特征 * 10个时间步
            )
        })

        # 思维链解释生成器
        self.explanation_generator = nn.ModuleDict({
            'feature_encoder': nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
            ),
            'current_state': nn.Linear(hidden_size // 2, 4),
            'trend_analyzer': nn.Linear(hidden_size // 2, 4),
            'risk_assessor': nn.Linear(hidden_size // 2, 3),
        })

    def analyze_trajectory_history(self, trajectory_features):
        """使用思维链方法分析轨迹历史"""
        # 计算速度变化
        velocities = trajectory_features[..., 2:4]
        speed = torch.norm(velocities, dim=-1)

        # 计算加速度
        acceleration = speed[1:] - speed[:-1]
        avg_acceleration = acceleration.mean().item()

        # 计算转向变化
        angles = torch.atan2(velocities[..., 1], velocities[..., 0])
        angle_changes = (angles[1:] - angles[:-1]).abs()
        avg_turning = angle_changes.mean().item()

        return {
            'avg_speed': speed.mean().item(),
            'speed_std': speed.std().item(),
            'avg_acceleration': avg_acceleration,
            'avg_turning': avg_turning
        }

    def compute_loss(self, predictions, targets):
        """计算多尺度预测损失，包含位置、速度和平滑度损失"""
        losses = {}
        total_loss = 0.0
        
        # 确保目标在正确的设备上
        targets = targets.to(self.device)
        
        # 1. 位置和速度损失
        # 短期预测损失 (4个时间步)
        short_term_target = targets[:, :4, :].contiguous()
        short_term_pred = predictions['short_term']
        
        # 分别计算位置和速度损失
        short_term_pos_loss = nn.MSELoss()(short_term_pred[..., :2], short_term_target[..., :2])
        short_term_vel_loss = nn.MSELoss()(short_term_pred[..., 2:], short_term_target[..., 2:])
        short_term_loss = short_term_pos_loss + 0.5 * short_term_vel_loss
        losses['short_term'] = short_term_loss.item()
        total_loss += 0.3 * short_term_loss

        # 中期预测损失 (6个时间步)
        mid_term_target = targets[:, :6, :].contiguous()
        mid_term_pred = predictions['mid_term']
        
        mid_term_pos_loss = nn.MSELoss()(mid_term_pred[..., :2], mid_term_target[..., :2])
        mid_term_vel_loss = nn.MSELoss()(mid_term_pred[..., 2:], mid_term_target[..., 2:])
        mid_term_loss = mid_term_pos_loss + 0.5 * mid_term_vel_loss
        losses['mid_term'] = mid_term_loss.item()
        total_loss += 0.3 * mid_term_loss

        # 长期预测损失 (10个时间步)
        long_term_target = targets[:, :10, :].contiguous()
        long_term_pred = predictions['long_term']
        
        long_term_pos_loss = nn.MSELoss()(long_term_pred[..., :2], long_term_target[..., :2])
        long_term_vel_loss = nn.MSELoss()(long_term_pred[..., 2:], long_term_target[..., 2:])
        long_term_loss = long_term_pos_loss + 0.5 * long_term_vel_loss
        losses['long_term'] = long_term_loss.item()
        total_loss += 0.4 * long_term_loss

        # 2. 轨迹平滑度损失
        def compute_smoothness_loss(pred):
            # 计算加速度（速度的差分）
            velocities = pred[..., 2:]  # 提取速度分量
            acceleration = velocities[:, 1:] - velocities[:, :-1]  # 计算加速度
            smoothness_loss = torch.mean(torch.square(acceleration))
            return smoothness_loss

        smoothness_loss = (
            0.3 * compute_smoothness_loss(short_term_pred) +
            0.3 * compute_smoothness_loss(mid_term_pred) +
            0.4 * compute_smoothness_loss(long_term_pred)
        )
        losses['smoothness'] = smoothness_loss.item()
        total_loss += 0.1 * smoothness_loss

        # 3. 速度连续性损失
        def compute_velocity_continuity_loss(pred, target):
            pred_vel_mag = torch.norm(pred[..., 2:], dim=-1)  # 预测速度大小
            target_vel_mag = torch.norm(target[..., 2:], dim=-1)  # 目标速度大小
            continuity_loss = nn.MSELoss()(pred_vel_mag, target_vel_mag)
            return continuity_loss

        velocity_continuity_loss = (
            0.3 * compute_velocity_continuity_loss(short_term_pred, short_term_target) +
            0.3 * compute_velocity_continuity_loss(mid_term_pred, mid_term_target) +
            0.4 * compute_velocity_continuity_loss(long_term_pred, long_term_target)
        )
        losses['velocity_continuity'] = velocity_continuity_loss.item()
        total_loss += 0.1 * velocity_continuity_loss

        # 4. L2正则化
        l2_lambda = 0.001  # 降低L2正则化强度
        l2_reg = torch.tensor(0., device=self.device)
        for param in self.parameters():
            if param.requires_grad:  # 只对可训练参数进行正则化
                l2_reg += torch.norm(param)
        l2_loss = l2_lambda * l2_reg
        losses['l2_reg'] = l2_loss.item()
        total_loss += l2_loss

        return total_loss, losses

    def generate_explanation_with_llm(self, prompt):
        """使用语言模型生成解释"""
        import time
        start_time = time.time()
        try:
            self.llm = self.llm.to(self.device)
            
            # 修改这里：增加max_length并使用max_new_tokens
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=1500  # 增加输入长度限制
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=500,  # 使用max_new_tokens替代max_length
                    num_return_sequences=1,   
                    temperature=0.8,         
                    top_p=0.95,             
                    top_k=100,              
                    do_sample=True,         
                    no_repeat_ngram_size=3, 
                    early_stopping=True,
                    min_length=150,         
                    repetition_penalty=1.2, 
                    length_penalty=1.5,     
                    num_beams=5            
                )
                outputs = outputs.cpu()
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 记录生成过程
            self.generation_tracker.log_generation(
                'llm',
                prompt,
                generated_text,
                time.time() - start_time
            )

            return generated_text

        except Exception as e:
            print(f"生成解释时出错: {str(e)}")
            return "生成解释失败"

    def forward(self, trajectory_features):
        batch_size = trajectory_features.shape[0]
        seq_len = trajectory_features.shape[1]

        # 特征归一化
        trajectory_features = trajectory_features.view(-1, 4)
        trajectory_features = self.feature_norm(trajectory_features)
        trajectory_features = trajectory_features.view(batch_size, seq_len, 4)

        # 空间特征提取
        spatial_features = self.trajectory_encoder['spatial'](trajectory_features)

        # 时间特征提取，使用LSTM
        temporal_features, (hidden, cell) = self.trajectory_encoder['temporal'](spatial_features)

        # 注意力机制
        attn_out, _ = self.trajectory_encoder['attention'](
            temporal_features.transpose(0, 1),
            temporal_features.transpose(0, 1),
            temporal_features.transpose(0, 1)
        )

        # 特征融合
        concat_features = torch.cat([
            attn_out[-1],
            hidden[-1],
            hidden[-2]
        ], dim=1)
        fused_features = self.trajectory_encoder['fusion'](concat_features)

        # 多尺度预测
        predictions = {
            'short_term': self.trajectory_predictors['short_term'](fused_features).view(batch_size, 4, 4),
            'mid_term': self.trajectory_predictors['mid_term'](fused_features).view(batch_size, 6, 4),
            'long_term': self.trajectory_predictors['long_term'](fused_features).view(batch_size, 10, 4)
        }

        # 使用思维链生成解释
        if not self.training or (hasattr(self, '_batch_count') and self._batch_count % 50 == 0):
            explanation = self.generate_explanation(
                trajectory_features[0], 
                fused_features[0:1],
                predictions
            )
        else:
            explanation = ""

        if hasattr(self, '_batch_count'):
            self._batch_count += 1
        else:
            self._batch_count = 1

        return predictions, explanation

    def generate_explanation(self, trajectory_features, hidden_state=None, predictions=None):
        """生成增强的思维链解释"""
        try:
            # 确保所有张量都在正确的设备上
            if isinstance(trajectory_features, torch.Tensor):
                trajectory_features = trajectory_features.detach().cpu().numpy()

            if hidden_state is not None and isinstance(hidden_state, torch.Tensor):
                hidden_state = hidden_state.detach().cpu().numpy()

            if len(trajectory_features.shape) == 3:
                trajectory_features = trajectory_features[0]

            # 将numpy数组转换为张量并移动到正确的设备
            trajectory_tensor = torch.from_numpy(trajectory_features).to(self.device)
            history_analysis = self.analyze_trajectory_history(trajectory_tensor)

            if hidden_state is not None:
                if len(hidden_state.shape) == 1:
                    hidden_state = hidden_state.reshape(1, -1)

                with torch.no_grad():
                    # 将hidden_state转换为张量并移动到正确的设备
                    hidden_tensor = torch.from_numpy(hidden_state).float().to(self.device)
                    
                    # 确保explanation_generator的所有组件都在正确的设备上
                    for module in self.explanation_generator.values():
                        module.to(self.device)
                    
                    # 生成特征
                    encoded_features = self.explanation_generator['feature_encoder'](hidden_tensor)
                    current_state = self.explanation_generator['current_state'](encoded_features)
                    trend_analysis = self.explanation_generator['trend_analyzer'](encoded_features)
                    risk_scores = torch.softmax(
                        self.explanation_generator['risk_assessor'](encoded_features),
                        dim=-1
                    )

                    # 将结果移到CPU并转换为numpy数组
                    current_state = current_state.cpu().numpy()
                    trend_analysis = trend_analysis.cpu().numpy()
                    risk_scores = risk_scores.cpu().numpy()

                return self._generate_enhanced_explanation(
                    current_state[0] if len(current_state.shape) > 1 else current_state,
                    trend_analysis[0] if len(trend_analysis.shape) > 1 else trend_analysis,
                    risk_scores[0] if len(risk_scores.shape) > 1 else risk_scores,
                    history_analysis,
                    predictions
                )

            return self._features_to_text(trajectory_features)

        except Exception as e:
            print(f"生成解释时出错: {str(e)}")
            return "生成解释失败"

    def _generate_enhanced_explanation(self, current_state, trend_analysis, risk_scores, history_analysis, predictions=None):
        """Generate explanation text using LLM"""
        try:
            current_speed = np.sqrt(current_state[2] ** 2 + current_state[3] ** 2)
            trend_speed = np.sqrt(trend_analysis[2] ** 2 + trend_analysis[3] ** 2)

            # Prepare state description
            if trend_speed > current_speed:
                speed_trend = "accelerating"
            elif trend_speed < current_speed * 0.9:
                speed_trend = "decelerating"
            else:
                speed_trend = "maintaining speed"

            current_angle = np.arctan2(current_state[3], current_state[2])
            trend_angle = np.arctan2(trend_analysis[3], trend_analysis[2])
            angle_diff = np.abs(trend_angle - current_angle)
            turning = "turning" if angle_diff > 0.3 else "moving straight"

            risk_levels = ["low", "medium", "high"]
            risk_level = risk_levels[np.argmax(risk_scores)]

            # Build prompt
            prompt = f"""Analyze the vehicle's trajectory and predict its future movement based on:
[Current State]
Position: ({current_state[0]:.2f}, {current_state[1]:.2f})
Current speed: {current_speed:.2f} m/s
Average speed: {history_analysis['avg_speed']:.2f} m/s
Acceleration: {history_analysis['avg_acceleration']:.2f} m/s²
Current motion: {speed_trend}, {turning}
Risk level: {risk_level}
"""

            # 如果有predictions，添加多尺度预测信息
            if predictions is not None:
                prompt += f"""
Short-term prediction (next 4 steps): {self._format_prediction(predictions['short_term'])}
Mid-term prediction (next 6 steps): {self._format_prediction(predictions['mid_term'])}
Long-term prediction (next 10 steps): {self._format_prediction(predictions['long_term'])}
"""

            return self.generate_explanation_with_llm(prompt)

        except Exception as e:
            print(f"Error generating explanation: {str(e)}")
            return "Failed to generate explanation"

    def _features_to_text(self, trajectory_features):
        """Convert numerical features to text description"""
        template = """Trajectory Analysis:
Position(x={:.2f}, y={:.2f}), Velocity(vx={:.2f}, vy={:.2f})
Analysis: Based on current position and velocity, vehicle is likely moving {direction}, expected to {speed_change}."""

        # Ensure we're working with numpy array
        if isinstance(trajectory_features, torch.Tensor):
            features = trajectory_features.detach().cpu().numpy()
        else:
            features = trajectory_features

        # Ensure we're working with 1D array
        if len(features.shape) > 1:
            features = features.flatten()[:4]

        # Analyze motion direction and speed changes
        vx, vy = features[2], features[3]
        speed = np.sqrt(vx ** 2 + vy ** 2)

        # Determine main motion direction
        if abs(vx) > abs(vy):
            direction = "to the " + ("right" if vx > 0 else "left")
        else:
            direction = ("forward" if vy > 0 else "backward")

        # Determine speed change
        if speed < 5:
            speed_change = "maintain low speed"
        elif speed < 15:
            speed_change = "maintain moderate speed"
        else:
            speed_change = "maintain high speed"

        return template.format(
            float(features[0]),
            float(features[1]),
            float(features[2]),
            float(features[3]),
            direction=direction,
            speed_change=speed_change
        )

    def train_llm_step(self, prompt, target_text):
        """训练LLM的单个步骤"""
        try:
            # 准备输入
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            targets = self.tokenizer(target_text, return_tensors="pt", padding=True, truncation=True)

            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            target_ids = targets["input_ids"].to(self.device)

            # 前向传播
            outputs = self.llm(**inputs, labels=target_ids)
            loss = outputs.loss

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.llm.parameters(), 1.0)

            # 优化器步进
            self.optimizer.step()
            self.optimizer.zero_grad()

            return loss.item()

        except Exception as e:
            print(f"Error in training step: {str(e)}")
            return None

    def train_llm(self, train_data, num_epochs=3, save_dir="lora_checkpoints"):
        """训练LLM"""
        os.makedirs(save_dir, exist_ok=True)
        best_loss = float('inf')

        try:
            for epoch in range(num_epochs):
                total_loss = 0
                num_batches = 0

                for prompt, target in train_data:
                    loss = self.train_llm_step(prompt, target)
                    if loss is not None:
                        total_loss += loss
                        num_batches += 1

                avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
                print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

                # 保存最佳模型
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.save_lora_weights(os.path.join(save_dir, "best_model"))

                # 定期保存
                if (epoch + 1) % 5 == 0:
                    self.save_lora_weights(os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}"))

        except Exception as e:
            print(f"Error in training: {str(e)}")
            # 保存最后的检查点
            self.save_lora_weights(os.path.join(save_dir, "last_checkpoint"))

    def save_lora_weights(self, path):
        """保存LoRA权重"""
        try:
            self.llm.save_pretrained(path)
            print(f"Saved LoRA weights to {path}")
        except Exception as e:
            print(f"Error saving LoRA weights: {str(e)}")

    def load_lora_weights(self, path):
        """加载LoRA权重"""
        try:
            self.llm = PeftModel.from_pretrained(self.llm, path)
            print(f"Loaded LoRA weights from {path}")
        except Exception as e:
            print(f"Error loading LoRA weights: {str(e)}")

    def prepare_training_data(self, trajectory_data):
        """准备训练数据"""
        training_pairs = []

        for trajectory in trajectory_data:
            # 生成输入提示词
            prompt = self._generate_prompt_from_trajectory(trajectory)

            # 生成目标文本（这里需要根据实际情况修改）
            target = self._generate_target_from_trajectory(trajectory)

            training_pairs.append((prompt, target))

        return training_pairs

    def _generate_prompt_from_trajectory(self, trajectory):
        """从轨迹生成提示词"""
        # 这里需要根据实际轨迹数据结构进行修改
        current_state = trajectory['current_state']
        history = trajectory['history']

        prompt = f"""Analyze the vehicle's trajectory and predict its future movement based on:
Position: ({current_state[0]:.2f}, {current_state[1]:.2f})
Current speed: {current_state[2]:.2f} m/s
History: {history}
"""
        return prompt

    def _generate_target_from_trajectory(self, trajectory):
        """从轨迹生成目标文本"""
        # 这里需要根据实际轨迹数据结构进行修改
        future_state = trajectory['future_state']

        target = f"""The vehicle is likely to:
1. Move towards position ({future_state[0]:.2f}, {future_state[1]:.2f})
2. Change speed to {future_state[2]:.2f} m/s
3. Follow a {future_state[3]} trajectory
"""
        return target

    def _format_prediction(self, prediction):
        """Format a prediction for display"""
        formatted_prediction = ""
        # prediction shape: (batch_size, time_steps, 4)
        # 只取第一个batch的数据，每个时间步取位置信息(x,y)
        time_steps = prediction.shape[1]  # 获取实际的时间步数
        for i in range(time_steps):
            formatted_prediction += f"({prediction[0, i, 0]:.2f}, {prediction[0, i, 1]:.2f}), "
        return formatted_prediction[:-2]  # 移除最后的逗号和空格
