# # import torch
# # import torch.nn as nn
# # from transformers import (
# #     AutoModelForCausalLM, 
# #     AutoTokenizer, 
# #     pipeline,
# #     GPT2LMHeadModel,
# #     GPT2Config,
# #     BertTokenizer,
# #     BertForMaskedLM
# # )
# # from peft import get_peft_model, LoraConfig, TaskType, PeftModel
# # import numpy as np
# # import json
# # from datetime import datetime
# # import os
# # import warnings
# # import re

# # class GenerationTracker:
# #     """跟踪LLM生成状态和模板使用情况"""
# #     def __init__(self, save_dir="generation_logs"):
# #         self.save_dir = save_dir
# #         os.makedirs(save_dir, exist_ok=True)
# #         self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
# #         self.stats = {
# #             'llm_generations': 0,
# #             'template_generations': 0,
# #             'failed_generations': 0,
# #             'generation_times': [],
# #             'prompts': [],
# #             'responses': [],
# #             'errors': []
# #         }
        
# #     def log_generation(self, generation_type, prompt, response, time_taken, error=None):
# #         """记录单次生成情况"""
# #         if generation_type == 'llm':
# #             self.stats['llm_generations'] += 1
# #         elif generation_type == 'template':
# #             self.stats['template_generations'] += 1
            
# #         if error:
# #             self.stats['failed_generations'] += 1
# #             self.stats['errors'].append({
# #                 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
# #                 'error': str(error),
# #                 'prompt': prompt
# #             })
        
# #         self.stats['generation_times'].append(time_taken)
# #         self.stats['prompts'].append(prompt)
# #         self.stats['responses'].append(response)
        
# #         # 定期保存统计信息
# #         if len(self.stats['generation_times']) % 100 == 0:
# #             self.save_stats()
    
# #     def get_summary(self):
# #         """获取生成统计摘要"""
# #         total_generations = self.stats['llm_generations'] + self.stats['template_generations']
# #         if total_generations == 0:
# #             return "尚无生成记录"
            
# #         avg_time = np.mean(self.stats['generation_times']) if self.stats['generation_times'] else 0
        
# #         return {
# #             '总生成次数': total_generations,
# #             'LLM生成次数': self.stats['llm_generations'],
# #             '模板生成次数': self.stats['template_generations'],
# #             '失败次数': self.stats['failed_generations'],
# #             '平均生成时间': f"{avg_time:.3f}秒",
# #             'LLM使用率': f"{(self.stats['llm_generations']/total_generations*100):.1f}%",
# #             '模板使用率': f"{(self.stats['template_generations']/total_generations*100):.1f}%",
# #             '成功率': f"{((total_generations-self.stats['failed_generations'])/total_generations*100):.1f}%"
# #         }
    
# #     def save_stats(self):
# #         """保存统计信息到文件"""
# #         stats_file = os.path.join(self.save_dir, f"generation_stats_{self.current_session}.json")
# #         with open(stats_file, 'w', encoding='utf-8') as f:
# #             json.dump({
# #                 'summary': self.get_summary(),
# #                 'detailed_stats': self.stats
# #             }, f, ensure_ascii=False, indent=2)

# # class CoTTrajectoryPredictor(nn.Module):
# #     def __init__(self, device=None, hidden_size=64):
# #         super().__init__()
# #         self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #         self.llm_model_name = "./bert-base-chinese"
        
# #         # 初始化生成跟踪器
# #         self.generation_tracker = GenerationTracker()

# #         print("正在加载语言模型...")
# #         try:
# #             self.tokenizer = BertTokenizer.from_pretrained(
# #                 self.llm_model_name,
# #                 local_files_only=True
# #             )
            
# #             # 加载BERT模型
# #             self.llm = BertForMaskedLM.from_pretrained(
# #                 self.llm_model_name,
# #                 torch_dtype=torch.float32,
# #                 low_cpu_mem_usage=True,
# #                 local_files_only=True
# #             )
            
# #             # 冻结BERT参数
# #             for param in self.llm.parameters():
# #                 param.requires_grad = False
            
# #             self.llm = self.llm.to(self.device)
# #             print(f"成功加载模型 {self.llm_model_name}")
# #         except Exception as e:
# #             print(f"加载模型时出错: {str(e)}")
# #             raise

# #         # 特征标准化层
# #         self.feature_norm = nn.BatchNorm1d(4, momentum=0.1)
        
# #         # 轨迹编码器
# #         self.trajectory_encoder = nn.ModuleDict({
# #             'spatial': nn.Sequential(
# #                 nn.Linear(4, hidden_size),
# #                 nn.ReLU(),
# #                 nn.Dropout(0.1)
# #             ),
# #             'temporal': nn.LSTM(  # 保持LSTM不变
# #                 input_size=hidden_size,
# #                 hidden_size=hidden_size,
# #                 num_layers=1,
# #                 batch_first=True,
# #                 dropout=0.1,
# #                 bidirectional=False
# #             ),
# #             'attention': nn.MultiheadAttention(
# #                 embed_dim=hidden_size,
# #                 num_heads=2,
# #                 dropout=0.1,
# #                 batch_first=True
# #             ),
# #             'fusion': nn.Sequential(
# #                 nn.Linear(hidden_size * 2, hidden_size),
# #                 nn.ReLU(),
# #                 nn.Dropout(0.1)
# #             )
# #         })
        
# #         # 轨迹预测器
# #         self.trajectory_predictors = nn.ModuleDict({
# #             'short_term': nn.Sequential(
# #                 nn.Linear(hidden_size, hidden_size),
# #                 nn.ReLU(),
# #                 nn.Dropout(0.1),
# #                 nn.Linear(hidden_size, 16)
# #             ),
# #             'mid_term': nn.Sequential(
# #                 nn.Linear(hidden_size, hidden_size),
# #                 nn.ReLU(),
# #                 nn.Dropout(0.1),
# #                 nn.Linear(hidden_size, 24)
# #             ),
# #             'long_term': nn.Sequential(
# #                 nn.Linear(hidden_size, hidden_size),
# #                 nn.ReLU(),
# #                 nn.Dropout(0.1),
# #                 nn.Linear(hidden_size, 40)
# #             )
# #         })
        
# #         # 初始化权重
# #         self._init_weights()
        
# #         print("模型初始化完成")
        
# #         # 打印可训练参数数量
# #         total_params = sum(p.numel() for p in self.parameters())
# #         trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
# #         print(f"可训练参数: {trainable_params:,d} / 总参数: {total_params:,d} / 可训练比例: {100 * trainable_params / total_params:.2f}%")

# #     def _init_weights(self):
# #         """初始化模型权重"""
# #         for name, module in self.named_modules():
# #             if isinstance(module, nn.Linear):
# #                 # 使用较小的初始值范围
# #                 nn.init.xavier_uniform_(module.weight, gain=0.01)
# #                 if module.bias is not None:
# #                     nn.init.zeros_(module.bias)
# #             elif isinstance(module, nn.LSTM):
# #                 for name, param in module.named_parameters():
# #                     if 'weight' in name:
# #                         nn.init.orthogonal_(param, gain=0.01)  # 使用较小的gain
# #                     elif 'bias' in name:
# #                         nn.init.zeros_(param)

# #     def analyze_trajectory_history(self, trajectory_features):
# #         """分析轨迹历史"""
# #         # 计算速度变化
# #         velocities = trajectory_features[..., 2:4]  # 提取速度分量
# #         speed = torch.norm(velocities, dim=-1)  # 计算速度大小
        
# #         # 计算加速度（速度变化）
# #         acceleration = speed[1:] - speed[:-1]
# #         avg_acceleration = acceleration.mean().item()
        
# #         # 计算转向变化
# #         angles = torch.atan2(velocities[..., 1], velocities[..., 0])
# #         angle_changes = (angles[1:] - angles[:-1]).abs()
# #         avg_turning = angle_changes.mean().item()
        
# #         return {
# #             'avg_speed': speed.mean().item(),
# #             'speed_std': speed.std().item(),
# #             'avg_acceleration': avg_acceleration,
# #             'avg_turning': avg_turning
# #         }
    
# #     def generate_explanation(self, trajectory_features, hidden_state=None):
# #         try:
# #             # 确保输入是在CPU上的numpy数组
# #             if isinstance(trajectory_features, torch.Tensor):
# #                 trajectory_features = trajectory_features.detach().cpu().numpy()

# #             if hidden_state is not None and isinstance(hidden_state, torch.Tensor):
# #                 hidden_state = hidden_state.detach().cpu().numpy()

# #             # 如果是3D张量，只取第一个序列
# #             if len(trajectory_features.shape) == 3:
# #                 trajectory_features = trajectory_features[0]

# #             # 分析历史轨迹
# #             history_analysis = self.analyze_trajectory_history(torch.from_numpy(trajectory_features))
            
# #             # 提取当前状态
# #             current_state = trajectory_features[-1]  # 使用最后一个时间步的状态
            
# #             # 计算趋势
# #             if len(trajectory_features) > 1:
# #                 velocity_diff = trajectory_features[-1, 2:4] - trajectory_features[-2, 2:4]
# #                 acceleration = np.sqrt(np.sum(velocity_diff ** 2))
# #                 turning = np.arctan2(velocity_diff[1], velocity_diff[0])
# #             else:
# #                 acceleration = 0
# #                 turning = 0
            
# #             # 计算风险分数
# #             current_speed = np.sqrt(np.sum(current_state[2:4] ** 2))
# #             risk_level = "高" if current_speed > 30 or abs(turning) > 0.5 else "中" if current_speed > 20 or abs(turning) > 0.3 else "低"
            
# #             # 生成解释
# #             return self._generate_enhanced_explanation(
# #                 current_state,
# #                 np.array([acceleration, turning]),
# #                 {"低": 0.2, "中": 0.3, "高": 0.5}[risk_level],
# #                 history_analysis
# #             )

# #         except Exception as e:
# #             print(f"生成解释时出错: {str(e)}")
# #             return "生成解释失败"

# #     def generate_explanation_with_pipeline(self, prompt):
# #         """使用BERT模型辅助生成解释"""
# #         import time
# #         start_time = time.time()
        
# #         try:
# #             # 从 prompt 中提取关键信息
# #             prompt = re.sub(r'[^\x00-\x7F]+', '', prompt)
# #             import re
            
# #             # 提取数值信息
# #             position_match = re.search(r'当前位置：\((.*?), (.*?)\)', prompt)
# #             speed = float(re.search(r'当前速度：(.*?) m/s', prompt).group(1))
# #             avg_speed = float(re.search(r'平均速度：(.*?) m/s', prompt).group(1))
# #             acceleration = float(re.search(r'加速度：(.*?) m/s²', prompt).group(1))
# #             turning = float(re.search(r'转向频率：(.*?)\n', prompt).group(1))
# #             trend_match = re.search(r'运动趋势：(.*?)，(.*?)\n', prompt)
# #             risk = re.search(r'风险等级：(.*?)\n', prompt).group(1)
            
# #             # 构建结构化提示
# #             template = f"""基于当前车辆状态分析：
# # 1. 位置和运动状态：
# # - 位置：({position_match.group(1)}, {position_match.group(2)})
# # - 趋势：{trend_match.group(1)}，{trend_match.group(2)}
# # - 转向：{'频繁变道' if turning > 0.5 else '保持直线行驶'}

# # 2. 速度分析：
# # - 当前：{speed:.2f} m/s
# # - 平均：{avg_speed:.2f} m/s
# # - 加速：{acceleration:.2f} m/s²
# # - 状态：{'加速中' if acceleration > 0 else '减速中' if acceleration < 0 else '速度平稳'}

# # 3. 风险：{risk}级
# # """
# #             # 使用BERT生成补充说明
# #             inputs = self.tokenizer(
# #                 template,
# #                 return_tensors="pt",
# #                 padding=True,
# #                 truncation=True,
# #                 max_length=128
# #             ).to(self.device)
            
# #             with torch.no_grad():
# #                 outputs = self.llm(**inputs)
# #                 logits = outputs.logits
                
# #                 # 获取每个位置最可能的token
# #                 predictions = torch.argmax(logits, dim=-1)
                
# #                 # 解码BERT的输出
# #                 bert_explanation = self.tokenizer.decode(predictions[0], skip_special_tokens=True)
            
# #             # 根据风险等级添加具体建议
# #             if risk == "高":
# #                 suggestions = """
# # 4. 驾驶建议：
# # - 立即降低车速，保持警惕
# # - 确保与前车保持足够安全距离
# # - 避免急转弯和突然变道
# # - 密切关注周围车辆动态
# # - 准备应对紧急情况"""
# #             elif risk == "中":
# #                 suggestions = """
# # 4. 驾驶建议：
# # - 适当调整车速，保持稳定
# # - 与前车保持合理距离
# # - 注意观察前方路况变化
# # - 避免不必要的变道
# # - 保持良好的行驶状态"""
# #             else:
# #                 suggestions = """
# # 4. 驾驶建议：
# # - 继续保持当前良好驾驶状态
# # - 定期观察周围交通环境
# # - 保持适度警惕性
# # - 适时调整速度和方向
# # - 为可能的突发情况做好准备"""

# #             # 组合最终解释
# #             final_explanation = template + bert_explanation + suggestions
            
# #             # 记录生成
# #             self.generation_tracker.log_generation(
# #                 'hybrid',
# #                 prompt,
# #                 final_explanation,
# #                 time.time() - start_time
# #             )
            
# #             # 返回生成的解释
# #             final_explanation = template + bert_explanation + suggestions
# #             final_explanation = re.sub(r'[^\x00-\x7F]+', '', final_explanation)
# #             return final_explanation
            
# #         except Exception as e:
# #             print(f"生成解释时出错: {str(e)}")
# #             return "无法解析车辆状态信息，请确保安全驾驶。"
# #             pass

# #     def forward(self, trajectory_features):
# #         """端到端预测流程"""
# #         batch_size = trajectory_features.shape[0]
# #         seq_len = trajectory_features.shape[1]
        
# #         # 检查输入数据
# #         if torch.isnan(trajectory_features).any():
# #             print("警告：输入特征包含NaN值")
# #             trajectory_features = torch.nan_to_num(trajectory_features, 0.0)
        
# #         # 特征标准化，添加eps防止除零
# #         trajectory_features = trajectory_features.view(-1, 4)
# #         trajectory_features = self.feature_norm(trajectory_features)
# #         trajectory_features = torch.clamp(trajectory_features, -10, 10)  # 限制范围
# #         trajectory_features = trajectory_features.view(batch_size, seq_len, 4)
        
# #         # 空间特征提取
# #         spatial_features = self.trajectory_encoder['spatial'](trajectory_features)
# #         spatial_features = torch.clamp(spatial_features, -10, 10)  # 限制范围
        
# #         # 时序特征提取
# #         temporal_features, (hidden, cell) = self.trajectory_encoder['temporal'](spatial_features)
# #         temporal_features = torch.clamp(temporal_features, -10, 10)  # 限制范围
        
# #         # 注意力机制
# #         attn_out, _ = self.trajectory_encoder['attention'](
# #             temporal_features,  # 移除transpose，因为我们添加了batch_first=True
# #             temporal_features,
# #             temporal_features,
# #             need_weights=False
# #         )
# #         attn_out = torch.clamp(attn_out, -10, 10)  # 限制范围
        
# #         # 特征融合
# #         hidden_final = hidden.squeeze(0)  # [batch, hidden_size]
# #         attn_final = attn_out[:, -1]  # 使用batch_first=True后的索引方式
# #         concat_features = torch.cat([attn_final, hidden_final], dim=1)
# #         fused_features = self.trajectory_encoder['fusion'](concat_features)
# #         fused_features = torch.clamp(fused_features, -10, 10)  # 限制范围
        
# #         # 多尺度预测
# #         predictions = {}
# #         for term, predictor in self.trajectory_predictors.items():
# #             pred = predictor(fused_features)
# #             if term == 'short_term':
# #                 pred = pred.view(batch_size, 4, 4)
# #             elif term == 'mid_term':
# #                 pred = pred.view(batch_size, 6, 4)
# #             else:  # long_term
# #                 pred = pred.view(batch_size, 10, 4)
# #             predictions[term] = torch.clamp(pred, -100, 100)  # 限制预测范围
        
# #         # 只在验证时或每50个batch生成一次解释
# #         if not self.training or (hasattr(self, '_batch_count') and self._batch_count % 50 == 0):
# #             explanation = self.generate_explanation(trajectory_features[0], fused_features)
# #         else:
# #             explanation = ""
            
# #         # 更新batch计数
# #         if hasattr(self, '_batch_count'):
# #             self._batch_count += 1
# #         else:
# #             self._batch_count = 1
        
# #         return predictions, explanation
    
# #     def compute_loss(self, predictions, targets, weights={'short_term': 0.4, 'mid_term': 0.3, 'long_term': 0.3}):
# #         """计算增强的多尺度预测损失"""
# #         total_loss = torch.tensor(0., device=targets.device, requires_grad=True)
# #         loss_components = {}
        
# #         # 检查输入
# #         if torch.isnan(targets).any():
# #             print("警告：目标数据包含NaN值")
# #             targets = torch.nan_to_num(targets, 0.0)
        
# #         # 计算全局统计量用于归一化
# #         all_targets = targets.reshape(-1, targets.shape[-1])
# #         target_mean = all_targets.mean(dim=0, keepdim=True)
# #         target_std = all_targets.std(dim=0, keepdim=True) + 1e-6  # 增加epsilon
        
# #         # 使用Huber损失代替MSE
# #         huber_loss = torch.nn.HuberLoss(reduction='mean', delta=1.0)
        
# #         for term, pred in predictions.items():
# #             if term == 'short_term':
# #                 term_targets = targets[:, :4, :]
# #             elif term == 'mid_term':
# #                 term_targets = targets[:, :6, :]
# #             else:  # long_term
# #                 term_targets = targets
            
# #             # 归一化预测和目标
# #             pred_norm = (pred - target_mean) / target_std
# #             target_norm = (term_targets - target_mean) / target_std
            
# #             # 使用Huber损失计算位置损失
# #             pos_loss = huber_loss(
# #                 pred_norm[..., :2], 
# #                 target_norm[..., :2]
# #             )
            
# #             # 使用Huber损失计算速度损失
# #             vel_loss = huber_loss(
# #                 pred_norm[..., 2:], 
# #                 target_norm[..., 2:]
# #             )
            
# #             # 添加平滑度损失
# #             if pred.size(1) > 1:
# #                 smooth_loss = huber_loss(
# #                     pred_norm[:, 1:] - pred_norm[:, :-1],
# #                     target_norm[:, 1:] - target_norm[:, :-1]
# #                 )
# #             else:
# #                 smooth_loss = torch.tensor(0., device=targets.device)
            
# #             # 组合损失，使用较小的权重
# #             term_loss = (
# #                 0.4 * weights[term] * pos_loss + 
# #                 0.3 * vel_loss + 
# #                 0.1 * smooth_loss
# #             )
            
# #             # 记录损失组件
# #             loss_components[f'{term}_pos'] = pos_loss.item()
# #             loss_components[f'{term}_vel'] = vel_loss.item()
# #             loss_components[f'{term}_smooth'] = smooth_loss.item()
            
# #             # 累加到总损失
# #             total_loss = total_loss + term_loss
        
# #         # 添加极小的L2正则化
# #         l2_lambda = 1e-7  # 减小L2正则化强度
# #         l2_reg = torch.tensor(0., device=targets.device)
# #         for param in self.parameters():
# #             if param.requires_grad:
# #                 l2_reg += torch.norm(param)
# #         reg_loss = l2_lambda * l2_reg
# #         total_loss = total_loss + reg_loss
# #         loss_components['reg_loss'] = reg_loss.item()
        
# #         # 检查损失值是否为nan
# #         if torch.isnan(total_loss):
# #             print("警告：损失值为NaN，返回零损失")
# #             return torch.tensor(0., device=targets.device, requires_grad=True), loss_components
        
# #         return total_loss, loss_components
    
# #     def _generate_enhanced_explanation(self, current_state, trend_analysis, risk_score, history_analysis):
# #         """生成增强的解释文本"""
# #         try:
# #             # 计算当前速度
# #             current_speed = np.sqrt(current_state[2]**2 + current_state[3]**2)
            
# #             # 分析趋势
# #             trend_desc = "加速" if trend_analysis[0] > 0 else "减速"
# #             turning_desc = "正在转向" if abs(trend_analysis[1]) > 0.5 else "直线行驶"
            
# #             # 风险评估
# #             risk_level = "高" if risk_score > 0.4 else "中" if risk_score > 0.2 else "低"
            
# #             prompt = f"""基于以下车辆状态生成行驶分析：
# # - 当前位置：({current_state[0]:.2f}, {current_state[1]:.2f})
# # - 当前速度：{current_speed:.2f} m/s
# # - 平均速度：{history_analysis['avg_speed']:.2f} m/s
# # - 加速度：{history_analysis['avg_acceleration']:.2f} m/s²
# # - 转向频率：{history_analysis['avg_turning']:.2f}
# # - 运动趋势：{trend_desc}，{turning_desc}
# # - 风险等级：{risk_level}

# # 请分析车辆状态并提供建议：
# # """
# #             return self.generate_explanation_with_pipeline(prompt)
        
# #         except Exception as e:
# #             print(f"生成增强解释时出错: {str(e)}")
# #             return "生成解释失败"
    
# #     def _features_to_text(self, trajectory_features):
# #         """使用模板生成文本描述"""
# #         import time
# #         start_time = time.time()
        
# #         try:
# #             # 确保是在 CPU 上并转换为 numpy
# #             if isinstance(trajectory_features, torch.Tensor):
# #                 features = trajectory_features.detach().cpu().numpy()
# #             else:
# #                 features = trajectory_features
                
# #             # 确保我们使用的是一维数组
# #             if len(features.shape) > 1:
# #                 features = features.flatten()[:4]  # 只取前4个值
                
# #             # 分析运动方向和速度变化
# #             vx, vy = features[2], features[3]
# #             speed = np.sqrt(vx**2 + vy**2)
            
# #             # 确定主要运动方向
# #             if abs(vx) > abs(vy):
# #                 direction = "向" + ("右" if vx > 0 else "左")
# #             else:
# #                 direction = "向" + ("前" if vy > 0 else "后")
                
# #             # 判断速度变化
# #             if speed < 5:
# #                 speed_change = "将保持低速行驶"
# #             elif speed < 15:
# #                 speed_change = "将保持中速行驶"
# #             else:
# #                 speed_change = "将保持高速行驶"
                
# #             template = """分析车辆轨迹：
# # 位置(x={:.2f}, y={:.2f}), 速度(vx={:.2f}, vy={:.2f})
# # 预测分析：基于当前位置和速度，车辆可能{}行驶，预计{}。"""
            
# #             result = template.format(
# #                 float(features[0]), float(features[1]),
# #                 float(features[2]), float(features[3]),
# #                 direction, speed_change
# #             )
            
# #             # 记录成功的模板生成
# #             self.generation_tracker.log_generation(
# #                 'template',
# #                 "特征转文本模板",
# #                 result,
# #                 time.time() - start_time
# #             )
            
# #             return result
            
# #         except Exception as e:
# #             # 记录失败的模板生成
# #             self.generation_tracker.log_generation(
# #                 'template',
# #                 "特征转文本模板",
# #                 "生成失败",
# #                 time.time() - start_time,
# #                 error=e
# #             )
# #             return "无法生成文本描述。"

# #     def get_generation_stats(self):
# #         """获取生成统计信息"""
# #         return self.generation_tracker.get_summary()

# #     def _generate_template_explanation(self, prompt):
# #         """备用的模板生成方法"""
# #         try:
# #             # 从 prompt 中提取关键信息
# #             import re
            
# #             # 提取数值信息
# #             position = re.findall(r'当前位置：\((.*?)\)', prompt)[0].split(', ')
# #             speed = float(re.findall(r'当前速度：(.*?) m/s', prompt)[0])
# #             avg_speed = float(re.findall(r'平均速度：(.*?) m/s', prompt)[0])
# #             acceleration = float(re.findall(r'加速度：(.*?) m/s²', prompt)[0])
# #             trend = re.findall(r'运动趋势：(.*?)，(.*?)\n', prompt)[0]
# #             risk = re.findall(r'风险等级：(.*?)\n', prompt)[0]
            
# #             # 生成中文解释
# #             explanation = f"""根据当前行驶状态分析：
# # 1. 位置状态：车辆当前位于坐标 ({position[0]}, {position[1]})
# # 2. 速度状态：当前速度 {speed:.2f} m/s，平均速度 {avg_speed:.2f} m/s
# # 3. 运动趋势：车辆正在{trend[0]}，{trend[1]}
# # 4. 风险评估：当前风险等级{risk}

# # 行驶建议：
# # """
# #             # 根据不同情况添加具体建议
# #             if risk == "高":
# #                 explanation += "- 请立即降低车速，保持安全距离\n- 谨慎驾驶，注意周围车辆状态\n- 避免急转弯和突然变道"
# #             elif risk == "中":
# #                 explanation += "- 建议适当调整车速\n- 保持正常行驶状态\n- 注意观察前方路况"
# #             else:
# #                 explanation += "- 当前行驶状态良好\n- 继续保持当前驾驶方式\n- 建议定期观察周围环境"
            
# #             return explanation
            
# #         except Exception as e:
# #             print(f"模板生成失败: {str(e)}")
# #             return "无法生成解释，请保持安全驾驶。"

# import torch
# import torch.nn as nn
# from transformers import (
#     AutoModelForCausalLM, 
#     AutoTokenizer, 
#     pipeline,
#     GPT2LMHeadModel,
#     GPT2Config,
#     BertTokenizer,
#     BertForMaskedLM
# )
# from peft import get_peft_model, LoraConfig, TaskType, PeftModel
# import numpy as np
# import json
# from datetime import datetime
# import os
# import warnings
# import re

# class GenerationTracker:
#     """跟踪LLM生成状态和模板使用情况"""
#     def __init__(self, save_dir="generation_logs"):
#         self.save_dir = save_dir
#         os.makedirs(save_dir, exist_ok=True)
#         self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.stats = {
#             'llm_generations': 0,
#             'template_generations': 0,
#             'hybrid_generations': 0,
#             'failed_generations': 0,
#             'generation_times': [],
#             'prompts': [],
#             'responses': [],
#             'errors': []
#         }
        
#     def log_generation(self, generation_type, prompt, response, time_taken, error=None):
#         """记录单次生成情况"""
#         if generation_type == 'llm':
#             self.stats['llm_generations'] += 1
#         elif generation_type == 'template':
#             self.stats['template_generations'] += 1
#         elif generation_type == 'hybrid':
#             self.stats['hybrid_generations'] += 1
            
#         if error:
#             self.stats['failed_generations'] += 1
#             self.stats['errors'].append({
#                 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                 'error': str(error),
#                 'prompt': prompt
#             })
        
#         self.stats['generation_times'].append(time_taken)
#         self.stats['prompts'].append(prompt)
#         self.stats['responses'].append(response)
        
#         # 每100次生成打印一次调试信息
#         if len(self.stats['generation_times']) % 100 == 0:
#             print(f"\n生成内容：\n{response}\n")

#         # 定期保存统计信息
#         if len(self.stats['generation_times']) % 100 == 0:
#             self.save_stats()
    
#     def get_summary(self):
#         """获取生成统计摘要"""
#         total_generations = (
#             self.stats['llm_generations'] +
#             self.stats['template_generations'] +
#             self.stats['hybrid_generations']
#         )
#         if total_generations == 0:
#             return "尚无生成记录"
            
#         avg_time = np.mean(self.stats['generation_times']) if self.stats['generation_times'] else 0
        
#         return {
#             '总生成次数': total_generations,
#             'LLM生成次数': self.stats['llm_generations'],
#             '模板生成次数': self.stats['template_generations'],
#             '混合生成次数': self.stats['hybrid_generations'],
#             '失败次数': self.stats['failed_generations'],
#             '平均生成时间': f"{avg_time:.3f}秒",
#             'LLM使用率': f"{(self.stats['llm_generations']/total_generations*100):.1f}%",
#             '模板使用率': f"{(self.stats['template_generations']/total_generations*100):.1f}%",
#             '混合使用率': f"{(self.stats['hybrid_generations']/total_generations*100):.1f}%",
#             '成功率': f"{((total_generations-self.stats['failed_generations'])/total_generations*100):.1f}%"
#         }
    
#     def save_stats(self):
#         """保存统计信息到文件"""
#         stats_file = os.path.join(self.save_dir, f"generation_stats_{self.current_session}.json")
#         with open(stats_file, 'w', encoding='utf-8') as f:
#             json.dump({
#                 'summary': self.get_summary(),
#                 'detailed_stats': self.stats
#             }, f, ensure_ascii=False, indent=2)

# class CoTTrajectoryPredictor(nn.Module):
#     def __init__(self, device=None, hidden_size=64):
#         super().__init__()
#         self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.llm_model_name = "./bert-base-chinese"
        
#         # 初始化生成跟踪器
#         self.generation_tracker = GenerationTracker()

#         print("正在加载语言模型...")
#         try:
#             self.tokenizer = BertTokenizer.from_pretrained(
#                 self.llm_model_name,
#                 local_files_only=True
#             )
            
#             # 加载BERT模型
#             self.llm = BertForMaskedLM.from_pretrained(
#                 self.llm_model_name,
#                 torch_dtype=torch.float32,
#                 low_cpu_mem_usage=True,
#                 local_files_only=True
#             )
            
#             # 冻结BERT参数
#             for param in self.llm.parameters():
#                 param.requires_grad = False
            
#             self.llm = self.llm.to(self.device)
#             print(f"成功加载模型 {self.llm_model_name}")
#         except Exception as e:
#             print(f"加载模型时出错: {str(e)}")
#             raise

#         # 特征标准化层
#         self.feature_norm = nn.BatchNorm1d(4, momentum=0.1)
        
#         # 轨迹编码器
#         self.trajectory_encoder = nn.ModuleDict({
#             'spatial': nn.Sequential(
#                 nn.Linear(4, hidden_size),
#                 nn.ReLU(),
#                 nn.Dropout(0.1)
#             ),
#             'temporal': nn.LSTM(  # 保持LSTM不变
#                 input_size=hidden_size,
#                 hidden_size=hidden_size,
#                 num_layers=1,
#                 batch_first=True,
#                 dropout=0.1,
#                 bidirectional=False
#             ),
#             'attention': nn.MultiheadAttention(
#                 embed_dim=hidden_size,
#                 num_heads=2,
#                 dropout=0.1,
#                 batch_first=True
#             ),
#             'fusion': nn.Sequential(
#                 nn.Linear(hidden_size * 2, hidden_size),
#                 nn.ReLU(),
#                 nn.Dropout(0.1)
#             )
#         })
        
#         # 轨迹预测器
#         self.trajectory_predictors = nn.ModuleDict({
#             'short_term': nn.Sequential(
#                 nn.Linear(hidden_size, hidden_size),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(hidden_size, 16)
#             ),
#             'mid_term': nn.Sequential(
#                 nn.Linear(hidden_size, hidden_size),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(hidden_size, 24)
#             ),
#             'long_term': nn.Sequential(
#                 nn.Linear(hidden_size, hidden_size),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(hidden_size, 40)
#             )
#         })
        
#         # 初始化权重
#         self._init_weights()
        
#         print("模型初始化完成")
        
#         # 打印可训练参数数量
#         total_params = sum(p.numel() for p in self.parameters())
#         trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
#         print(f"可训练参数: {trainable_params:,d} / 总参数: {total_params:,d} / 可训练比例: {100 * trainable_params / total_params:.2f}%")

#     def _init_weights(self):
#         """初始化模型权重"""
#         for name, module in self.named_modules():
#             if isinstance(module, nn.Linear):
#                 # 使用较小的初始值范围
#                 nn.init.xavier_uniform_(module.weight, gain=0.01)
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)
#             elif isinstance(module, nn.LSTM):
#                 for name, param in module.named_parameters():
#                     if 'weight' in name:
#                         nn.init.orthogonal_(param, gain=0.01)  # 使用较小的gain
#                     elif 'bias' in name:
#                         nn.init.zeros_(param)

#     def analyze_trajectory_history(self, trajectory_features):
#         """分析轨迹历史"""
#         # 计算速度变化
#         velocities = trajectory_features[..., 2:4]  # 提取速度分量
#         speed = torch.norm(velocities, dim=-1)  # 计算速度大小
        
#         # 计算加速度（速度变化）
#         acceleration = speed[1:] - speed[:-1]
#         avg_acceleration = acceleration.mean().item()
        
#         # 计算转向变化
#         angles = torch.atan2(velocities[..., 1], velocities[..., 0])
#         angle_changes = (angles[1:] - angles[:-1]).abs()
#         avg_turning = angle_changes.mean().item()
        
#         return {
#             'avg_speed': speed.mean().item(),
#             'speed_std': speed.std().item(),
#             'avg_acceleration': avg_acceleration,
#             'avg_turning': avg_turning
#         }
    
#     def generate_explanation(self, trajectory_features, hidden_state=None):
#         try:
#             # 确保输入是在CPU上的numpy数组
#             if isinstance(trajectory_features, torch.Tensor):
#                 trajectory_features = trajectory_features.detach().cpu().numpy()

#             if hidden_state is not None and isinstance(hidden_state, torch.Tensor):
#                 hidden_state = hidden_state.detach().cpu().numpy()

#             # 如果是3D张量，只取第一个序列
#             if len(trajectory_features.shape) == 3:
#                 trajectory_features = trajectory_features[0]

#             # 分析历史轨迹
#             history_analysis = self.analyze_trajectory_history(torch.from_numpy(trajectory_features))
            
#             # 提取当前状态
#             current_state = trajectory_features[-1]  # 使用最后一个时间步的状态
            
#             # 计算趋势
#             if len(trajectory_features) > 1:
#                 velocity_diff = trajectory_features[-1, 2:4] - trajectory_features[-2, 2:4]
#                 acceleration = np.sqrt(np.sum(velocity_diff ** 2))
#                 turning = np.arctan2(velocity_diff[1], velocity_diff[0])
#             else:
#                 acceleration = 0
#                 turning = 0
            
#             # 计算风险分数
#             current_speed = np.sqrt(np.sum(current_state[2:4] ** 2))
#             risk_level = "高" if current_speed > 30 or abs(turning) > 0.5 else "中" if current_speed > 20 or abs(turning) > 0.3 else "低"
            
#             # 生成解释
#             return self._generate_enhanced_explanation(
#                 current_state,
#                 np.array([acceleration, turning]),
#                 {"低": 0.2, "中": 0.3, "高": 0.5}[risk_level],
#                 history_analysis
#             )

#         except Exception as e:
#             print(f"生成解释时出错: {str(e)}")
#             return "生成解释失败"
#     def generate_explanation_with_pipeline(self, prompt):
# #     """使用BERT模型辅助生成解释"""
#         import time
#         start_time = time.time()

#         try:
#             # 打印 prompt 调试
#             print("Prompt:", prompt)

#             # 提取数值信息
#             import re

#             position_match = re.search(r'当前位置：\((.*?), (.*?)\)', prompt)
#             if not position_match:
#                 raise ValueError("无法匹配位置信息，请检查 prompt 格式")

#             speed_match = re.search(r'当前速度：(.*?) m/s', prompt)
#             if not speed_match:
#                 raise ValueError("无法匹配当前速度信息，请检查 prompt 格式")
#             speed = float(speed_match.group(1))

#             avg_speed_match = re.search(r'平均速度：(.*?) m/s', prompt)
#             if not avg_speed_match:
#                 raise ValueError("无法匹配平均速度信息，请检查 prompt 格式")
#             avg_speed = float(avg_speed_match.group(1))

#             acceleration_match = re.search(r'加速度：(.*?) m/s²', prompt)
#             if not acceleration_match:
#                 raise ValueError("无法匹配加速度信息，请检查 prompt 格式")
#             acceleration = float(acceleration_match.group(1))

#             turning_match = re.search(r'转向频率：(.*?)\n', prompt)
#             if not turning_match:
#                 raise ValueError("无法匹配转向频率信息，请检查 prompt 格式")
#             turning = float(turning_match.group(1))

#             trend_match = re.search(r'运动趋势：(.*?)，(.*?)\n', prompt)
#             if not trend_match:
#                 raise ValueError("无法匹配运动趋势信息，请检查 prompt 格式")

#             risk_match = re.search(r'风险等级：(.*?)\n', prompt)
#             if not risk_match:
#                 raise ValueError("无法匹配风险等级信息，请检查 prompt 格式")
#             risk = risk_match.group(1)

#             # 构建结构化提示
#             template = f"""基于当前车辆状态分析：
#     1. 位置和运动状态：
#     - 位置：({position_match.group(1)}, {position_match.group(2)})
#     - 趋势：{trend_match.group(1)}，{trend_match.group(2)}
#     - 转向：{'频繁变道' if turning > 0.5 else '保持直线行驶'}

#     2. 速度分析：
#     - 当前：{speed:.2f} m/s
#     - 平均：{avg_speed:.2f} m/s
#     - 加速：{acceleration:.2f} m/s²
#     - 状态：{'加速中' if acceleration > 0 else '减速中' if acceleration < 0 else '速度平稳'}

#     3. 风险：{risk}级
#     """
#             # 使用BERT生成补充说明
#             inputs = self.tokenizer(
#                 template,
#                 return_tensors="pt",
#                 padding=True,
#                 truncation=True,
#                 max_length=128
#             ).to(self.device)

#             with torch.no_grad():
#                 outputs = self.llm(**inputs)
#                 logits = outputs.logits
#                 predictions = torch.argmax(logits, dim=-1)
#                 bert_explanation = self.tokenizer.decode(predictions[0], skip_special_tokens=True)

#             # 根据风险等级添加具体建议
#             if risk == "高":
#                 suggestions = """4. 驾驶建议：
#     - 立即降低车速，保持警惕
#     - 确保与前车保持足够安全距离
#     - 避免急转弯和突然变道
#     - 密切关注周围车辆动态
#     - 准备应对紧急情况"""
#             elif risk == "中":
#                 suggestions = """4. 驾驶建议：
#     - 适当调整车速，保持稳定
#     - 与前车保持合理距离
#     - 注意观察前方路况变化
#     - 避免不必要的变道
#     - 保持良好的行驶状态"""
#             else:
#                 suggestions = """4. 驾驶建议：
#     - 继续保持当前良好驾驶状态
#     - 定期观察周围交通环境
#     - 保持适度警惕性
#     - 适时调整速度和方向
#     - 为可能的突发情况做好准备"""

#             # 组合最终解释
#             final_explanation = template + bert_explanation + suggestions

#             # 记录生成
#             self.generation_tracker.log_generation(
#                 'hybrid',
#                 prompt,
#                 final_explanation,
#                 time.time() - start_time
#             )
#             return final_explanation

#         except Exception as e:
#             print(f"生成解释时出错: {str(e)}")
#             return "无法解析车辆状态信息，请确保安全驾驶。"
# #     def generate_explanation_with_pipeline(self, prompt):
# #         """使用BERT模型辅助生成解释"""
# #         import time
# #         start_time = time.time()
        
# #         try:
# #             # 从 prompt 中提取关键信息
# #             prompt = re.sub(r'[^\x00-\x7F]+', '', prompt)
          
            
# #             # 提取数值信息
# #             position_match = re.search(r'当前位置：\((.*?), (.*?)\)', prompt)
# #             speed = float(re.search(r'当前速度：(.*?) m/s', prompt).group(1))
# #             avg_speed = float(re.search(r'平均速度：(.*?) m/s', prompt).group(1))
# #             acceleration = float(re.search(r'加速度：(.*?) m/s²', prompt).group(1))
# #             turning = float(re.search(r'转向频率：(.*?)\n', prompt).group(1))
# #             trend_match = re.search(r'运动趋势：(.*?)，(.*?)\n', prompt)
# #             risk = re.search(r'风险等级：(.*?)\n', prompt).group(1)
            
# #             # 构建结构化提示
# #             template = f"""基于当前车辆状态分析：
# # 1. 位置和运动状态：
# # - 位置：({position_match.group(1)}, {position_match.group(2)})
# # - 趋势：{trend_match.group(1)}，{trend_match.group(2)}
# # - 转向：{'频繁变道' if turning > 0.5 else '保持直线行驶'}

# # 2. 速度分析：
# # - 当前：{speed:.2f} m/s
# # - 平均：{avg_speed:.2f} m/s
# # - 加速：{acceleration:.2f} m/s²
# # - 状态：{'加速中' if acceleration > 0 else '减速中' if acceleration < 0 else '速度平稳'}

# # 3. 风险等级：{risk}
# # """
# #             # 使用BERT生成补充说明
# #             inputs = self.tokenizer(
# #                 template,
# #                 return_tensors="pt",
# #                 padding=True,
# #                 truncation=True,
# #                 max_length=128
# #             ).to(self.device)
            
# #             with torch.no_grad():
# #                 outputs = self.llm(**inputs)
# #                 logits = outputs.logits
# #                 # 获取每个位置最可能的token
# #                 predictions = torch.argmax(logits, dim=-1)
# #                 # 解码BERT的输出
# #                 bert_explanation = self.tokenizer.decode(predictions[0], skip_special_tokens=True)
            
# #             # 根据风险等级添加具体建议
# #             if risk == "高":
# #                 suggestions = """4. 驾驶建议：
# # - 立即降低车速，保持警惕
# # - 确保与前车保持足够安全距离
# # - 避免急转弯和突然变道
# # - 密切关注周围车辆动态
# # - 准备应对紧急情况"""
# #             elif risk == "中":
# #                 suggestions = """4. 驾驶建议：
# # - 适当调整车速，保持稳定
# # - 与前车保持合理距离
# # - 注意观察前方路况变化
# # - 避免不必要的变道
# # - 保持良好的行驶状态"""
# #             else:
# #                 suggestions = """4. 驾驶建议：
# # - 继续保持当前良好驾驶状态
# # - 定期观察周围交通环境
# # - 保持适度警惕性
# # - 适时调整速度和方向
# # - 为可能的突发情况做好准备"""

# #             # 组合最终解释
# #             final_explanation = template + bert_explanation + suggestions
            
# #             # 清理生成的文本
# #             final_explanation = re.sub(r'[^\x00-\x7F\u4e00-\u9fa5]+', '', final_explanation)
            
# #             # 记录生成
# #             self.generation_tracker.log_generation(
# #                 'hybrid',
# #                 prompt,
# #                 final_explanation,
# #                 time.time() - start_time
# #             )
# #             return final_explanation
            
# #         except Exception as e:
# #             print(f"生成解释时出错: {str(e)}")
# #             return "无法解析车辆状态信息，请确保安全驾驶。"

#     def forward(self, trajectory_features):
#         """端到端预测流程"""
#         batch_size = trajectory_features.shape[0]
#         seq_len = trajectory_features.shape[1]
        
#         # 检查输入数据
#         if torch.isnan(trajectory_features).any():
#             print("警告：输入特征包含NaN值")
#             trajectory_features = torch.nan_to_num(trajectory_features, 0.0)
        
#         # 特征标准化，添加eps防止除零
#         trajectory_features = trajectory_features.view(-1, 4)
#         trajectory_features = self.feature_norm(trajectory_features)
#         trajectory_features = torch.clamp(trajectory_features, -10, 10)  # 限制范围
#         trajectory_features = trajectory_features.view(batch_size, seq_len, 4)
        
#         # 空间特征提取
#         spatial_features = self.trajectory_encoder['spatial'](trajectory_features)
#         spatial_features = torch.clamp(spatial_features, -10, 10)  # 限制范围
        
#         # 时序特征提取
#         temporal_features, (hidden, cell) = self.trajectory_encoder['temporal'](spatial_features)
#         temporal_features = torch.clamp(temporal_features, -10, 10)  # 限制范围
        
#         # 注意力机制
#         attn_out, _ = self.trajectory_encoder['attention'](
#             temporal_features,  # 移除transpose，因为我们添加了batch_first=True
#             temporal_features,
#             temporal_features,
#             need_weights=False
#         )
#         attn_out = torch.clamp(attn_out, -10, 10)  # 限制范围
        
#         # 特征融合
#         hidden_final = hidden.squeeze(0)  # [batch, hidden_size]
#         attn_final = attn_out[:, -1]  # 使用batch_first=True后的索引方式
#         concat_features = torch.cat([attn_final, hidden_final], dim=1)
#         fused_features = self.trajectory_encoder['fusion'](concat_features)
#         fused_features = torch.clamp(fused_features, -10, 10)  # 限制范围
        
#         # 多尺度预测
#         predictions = {}
#         for term, predictor in self.trajectory_predictors.items():
#             pred = predictor(fused_features)
#             if term == 'short_term':
#                 pred = pred.view(batch_size, 4, 4)
#             elif term == 'mid_term':
#                 pred = pred.view(batch_size, 6, 4)
#             else:  # long_term
#                 pred = pred.view(batch_size, 10, 4)
#             predictions[term] = torch.clamp(pred, -100, 100)  # 限制预测范围
        
#         # 只在验证时或每50个batch生成一次解释
#         if not self.training or (hasattr(self, '_batch_count') and self._batch_count % 50 == 0):
#             explanation = self.generate_explanation(trajectory_features[0], fused_features)
#         else:
#             explanation = ""
            
#         # 更新batch计数
#         if hasattr(self, '_batch_count'):
#             self._batch_count += 1
#         else:
#             self._batch_count = 1
        
#         return predictions, explanation
    
#     def compute_loss(self, predictions, targets, weights={'short_term': 0.4, 'mid_term': 0.3, 'long_term': 0.3}):
#         """计算增强的多尺度预测损失"""
#         total_loss = torch.tensor(0., device=targets.device, requires_grad=True)
#         loss_components = {}
        
#         # 检查输入
#         if torch.isnan(targets).any():
#             print("警告：目标数据包含NaN值")
#             targets = torch.nan_to_num(targets, 0.0)
        
#         # 计算全局统计量用于归一化
#         all_targets = targets.reshape(-1, targets.shape[-1])
#         target_mean = all_targets.mean(dim=0, keepdim=True)
#         target_std = all_targets.std(dim=0, keepdim=True) + 1e-6  # 增加epsilon
        
#         # 使用Huber损失代替MSE
#         huber_loss = torch.nn.HuberLoss(reduction='mean', delta=1.0)
        
#         for term, pred in predictions.items():
#             if term == 'short_term':
#                 term_targets = targets[:, :4, :]
#             elif term == 'mid_term':
#                 term_targets = targets[:, :6, :]
#             else:  # long_term
#                 term_targets = targets
            
#             # 归一化预测和目标
#             pred_norm = (pred - target_mean) / target_std
#             target_norm = (term_targets - target_mean) / target_std
            
#             # 使用Huber损失计算位置损失
#             pos_loss = huber_loss(
#                 pred_norm[..., :2], 
#                 target_norm[..., :2]
#             )
            
#             # 使用Huber损失计算速度损失
#             vel_loss = huber_loss(
#                 pred_norm[..., 2:], 
#                 target_norm[..., 2:]
#             )
            
#             # 添加平滑度损失
#             if pred.size(1) > 1:
#                 smooth_loss = huber_loss(
#                     pred_norm[:, 1:] - pred_norm[:, :-1],
#                     target_norm[:, 1:] - target_norm[:, :-1]
#                 )
#             else:
#                 smooth_loss = torch.tensor(0., device=targets.device)
            
#             # 组合损失，使用较小的权重
#             term_loss = (
#                 0.4 * weights[term] * pos_loss + 
#                 0.3 * vel_loss + 
#                 0.1 * smooth_loss
#             )
            
#             # 记录损失组件
#             loss_components[f'{term}_pos'] = pos_loss.item()
#             loss_components[f'{term}_vel'] = vel_loss.item()
#             loss_components[f'{term}_smooth'] = smooth_loss.item()
            
#             # 累加到总损失
#             total_loss = total_loss + term_loss
        
#         # 添加极小的L2正则化
#         l2_lambda = 1e-7  # 减小L2正则化强度
#         l2_reg = torch.tensor(0., device=targets.device)
#         for param in self.parameters():
#             if param.requires_grad:
#                 l2_reg += torch.norm(param)
#         reg_loss = l2_lambda * l2_reg
#         total_loss = total_loss + reg_loss
#         loss_components['reg_loss'] = reg_loss.item()
        
#         # 检查损失值是否为nan
#         if torch.isnan(total_loss):
#             print("警告：损失值为NaN，返回零损失")
#             return torch.tensor(0., device=targets.device, requires_grad=True), loss_components
        
#         return total_loss, loss_components
    
#     def _generate_enhanced_explanation(self, current_state, trend_analysis, risk_score, history_analysis):
#         """生成增强的解释文本"""
#         try:
#             # 计算当前速度
#             current_speed = np.sqrt(current_state[2]**2 + current_state[3]**2)
            
#             # 分析趋势
#             trend_desc = "加速" if trend_analysis[0] > 0 else "减速"
#             turning_desc = "正在转向" if abs(trend_analysis[1]) > 0.5 else "直线行驶"
            
#             # 风险评估
#             risk_level = "高" if risk_score > 0.4 else "中" if risk_score > 0.2 else "低"
            
#             prompt = f"""基于以下车辆状态生成行驶分析：
# - 当前位置：({current_state[0]:.2f}, {current_state[1]:.2f})
# - 当前速度：{current_speed:.2f} m/s
# - 平均速度：{history_analysis['avg_speed']:.2f} m/s
# - 加速度：{history_analysis['avg_acceleration']:.2f} m/s²
# - 转向频率：{history_analysis['avg_turning']:.2f}
# - 运动趋势：{trend_desc}，{turning_desc}
# - 风险等级：{risk_level}

# 请分析车辆状态并提供建议："""
#             return self.generate_explanation_with_pipeline(prompt)
        
#         except Exception as e:
#             print(f"生成增强解释时出错: {str(e)}")
#             return "生成解释失败"

#     def _features_to_text(self, trajectory_features):
#         """使用模板生成文本描述"""
#         import time
#         start_time = time.time()
        
#         try:
#             # 确保是在 CPU 上并转换为 numpy
#             if isinstance(trajectory_features, torch.Tensor):
#                 features = trajectory_features.detach().cpu().numpy()
#             else:
#                 features = trajectory_features
#             if len(features.shape) > 1:
#                 features = features.flatten()[:4]  # 只取前4个值
#             vx, vy = features[2], features[3]
#             speed = np.sqrt(vx**2 + vy**2)
            
#             # 确定主要运动方向
#             direction = "向右" if vx > 0 else "向左" if abs(vx) > abs(vy) else "向前" if vy > 0 else "向后"
#             speed_level = "低速" if speed < 5 else "中速" if speed < 15 else "高速"
            
#             template = """分析车辆轨迹：
# 位置(x={:.2f}, y={:.2f})
# 速度(vx={:.2f}, vy={:.2f}) - 速度等级：{}
# 预测分析：基于当前位置和速度，车辆可能{}行驶，预计{}行驶。"""
            
#             result = template.format(
#                 float(features[0]), float(features[1]),
#                 float(vx), float(vy),
#                 speed_level,
#                 direction,
#                 speed_level
#             )
            
#             # 记录生成
#             self.generation_tracker.log_generation(
#                 'template',
#                 "特征转文本模板",
#                 result,
#                 time.time() - start_time
#             )
#             return result
            
#         except Exception as e:
#             self.generation_tracker.log_generation(
#                 'template',
#                 "特征转文本模板",
#                 "生成失败",
#                 time.time() - start_time,
#                 error=e
#             )
#             return "无法生成文本描述。"

#     def get_generation_stats(self):
#         """获取生成统计信息"""
#         return self.generation_tracker.get_summary()

#     def _generate_template_explanation(self, prompt):
#         """备用的模板生成方法"""
#         try:
#             # 提取数值信息
#             position = re.findall(r'当前位置：\((.*?)\)', prompt)[0].split(', ')
#             speed = float(re.findall(r'当前速度：(.*?) m/s', prompt)[0])
#             avg_speed = float(re.findall(r'平均速度：(.*?) m/s', prompt)[0])
#             acceleration = float(re.findall(r'加速度：(.*?) m/s²', prompt)[0])
#             trend = re.findall(r'运动趋势：(.*?)，(.*?)\n', prompt)[0]
#             risk = re.findall(r'风险等级：(.*?)\n', prompt)[0]
            
#             # 生成中文解释
#             explanation = f"""当前行驶状态分析：
# 位置：\n({position[0]}, {position[1]})
# 速度：当前 {speed:.2f} m/s，平均 {avg_speed:.2f} m/s  
# 加速度：{acceleration:.2f} m/s²  
# 运动趋势：{trend[0]}，{trend[1]}  
# 风险等级：{risk}

# 驾驶建议："""
            
#             # 根据风险等级添加建议
#             if risk == "高":
#                 explanation += """
# - 立即减速，确保安全距离
# - 避免急转弯和突然变道
# - 密切关注周围车辆动态"""
#             elif risk == "中":
#                 explanation += """
# - 适当调整车速
# - 保持合理车距
# - 注意观察路况"""
#             else:
#                 explanation += """
# - 继续保持良好状态
# - 定期观察周围环境
# - 维持适度警惕"""

#             return explanation
            
#         except Exception as e:
#             print(f"模板生成失败: {str(e)}")
#             return "无法生成解释，请保持安全驾驶。"





















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
            # 第1步：确保模型在GPU上
            self.llm = self.llm.to(self.device)  # self.device 通常是 cuda:0
            
            # 第2步：将输入数据移到GPU上
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}  # 移到GPU

            # 第3步：在GPU上生成文本
            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    max_length=500,          # 增加最大长度，允许生成更长的解释
                    num_return_sequences=1,   
                    temperature=0.8,         # 略微增加温度，使生成更有创造性
                    top_p=0.95,             # 增加top_p，考虑更多可能性
                    top_k=100,              # 增加top_k，考虑更多token选择
                    do_sample=True,         
                    no_repeat_ngram_size=3, # 增加n-gram大小，避免更长的重复
                    early_stopping=True,
                    min_length=150,         # 添加最小长度，确保生成足够详细
                    repetition_penalty=1.2, # 添加重复惩罚，避免重复内容
                    length_penalty=1.5,     # 添加长度奖励，鼓励生成更长的解释
                    num_beams=5            # 使用束搜索，生成更连贯的文本
                )  # outputs 在 GPU 上
                
#                 outputs = self.llm.generate(
#                     **inputs,
#                     max_length=200,
#                     num_return_sequences=1,
#                     temperature=0.7,
#                     top_p=0.9,
#                     top_k=50,
#                     do_sample=True,
#                     no_repeat_ngram_size=2,
#                     early_stopping=True
#                 )  # outputs 在 GPU 上
                
    
                # 第4步：将输出移到CPU上以进行解码
                # 因为tokenizer的decode操作必须在CPU上进行
                outputs = outputs.cpu()

            # 第5步：在CPU上进行解码
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
        """端到端预测流程，包含LSTM和注意力机制"""
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
            explanation = self.generate_explanation(trajectory_features[0], fused_features[0:1])
        else:
            explanation = ""

        if hasattr(self, '_batch_count'):
            self._batch_count += 1
        else:
            self._batch_count = 1

        return predictions, explanation

    def generate_explanation(self, trajectory_features, hidden_state=None):
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
                    history_analysis
                )

            return self._features_to_text(trajectory_features)

        except Exception as e:
            print(f"生成解释时出错: {str(e)}")
            return "生成解释失败"

    def _generate_enhanced_explanation(self, current_state, trend_analysis, risk_scores, history_analysis):
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
Position: ({current_state[0]:.2f}, {current_state[1]:.2f})
Current speed: {current_speed:.2f} m/s
Average speed: {history_analysis['avg_speed']:.2f} m/s
Acceleration: {history_analysis['avg_acceleration']:.2f} m/s²
Current motion: {speed_trend}, {turning}
Risk level: {risk_level}
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
