"""
训练模块
负责模型训练、验证和保存
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import logging

from models import MultiModalTransformerRecommender
from evaluator import evaluate_model, RecommenderEvaluator
from config import Config


class Trainer:
    """训练器类"""

    def __init__(self, model, train_loader, val_loader, config=None):
        self.config = config if config else Config()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = self.config.device

        # 将模型移动到设备
        self.model.to(self.device)

        # 损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

        # 早停机制
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # 记录训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_metrics_history = []
        self.val_metrics_history = []

        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.config.log_dir)

        # 评估器
        self.evaluator = RecommenderEvaluator(self.config.top_k_list)

        # 初始化日志记录器
        self.setup_logger()

    def setup_logger(self):
        """设置日志记录器"""
        # 获取当前时间
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建日志文件名：模型_数据集_时间.txt
        model_name = "MultiModalTransformer"
        dataset_name = f"Synthetic_{self.config.num_users}users_{self.config.num_items}items"
        log_filename = f"{model_name}_{dataset_name}_{current_time}.txt"

        # 确保logs目录存在
        os.makedirs(self.config.log_dir, exist_ok=True)

        # 设置日志文件路径
        self.log_file_path = os.path.join(self.config.log_dir, log_filename)

        # 配置日志记录器
        self.logger = logging.getLogger('TrainingLogger')
        self.logger.setLevel(logging.INFO)

        # 清除已有的处理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # 创建文件处理器
        file_handler = logging.FileHandler(self.log_file_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 创建格式器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加处理器到日志记录器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # 记录训练开始信息
        self.log_training_header()

    def log_training_header(self):
        """记录训练头部信息"""
        self.logger.info("=" * 80)
        self.logger.info("基于Transformer的多模态推荐系统 - 训练日志")
        self.logger.info("=" * 80)
        self.logger.info(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("")

        # 模型信息
        self.logger.info("模型配置:")
        self.logger.info(f"  模型名称: MultiModalTransformerRecommender")
        self.logger.info(f"  嵌入维度: {self.config.embedding_dim}")
        self.logger.info(f"  隐藏维度: {self.config.hidden_dim}")
        self.logger.info(f"  注意力头数: {self.config.num_heads}")
        self.logger.info(f"  Transformer层数: {self.config.num_layers}")
        self.logger.info(f"  Dropout率: {self.config.dropout}")
        self.logger.info("")

        # 数据集信息
        self.logger.info("数据集配置:")
        self.logger.info(f"  数据集类型: 合成数据集")
        self.logger.info(f"  用户数量: {self.config.num_users:,}")
        self.logger.info(f"  物品数量: {self.config.num_items:,}")
        self.logger.info(f"  交互数量: {self.config.num_interactions:,}")
        self.logger.info(f"  稀疏度: {self.config.sparsity:.2%}")
        self.logger.info("")

        # 训练配置
        self.logger.info("训练配置:")
        self.logger.info(f"  批次大小: {self.config.batch_size}")
        self.logger.info(f"  学习率: {self.config.learning_rate}")
        self.logger.info(f"  权重衰减: {self.config.weight_decay}")
        self.logger.info(f"  训练轮次: {self.config.num_epochs}")
        self.logger.info(f"  早停耐心值: {self.config.patience}")
        self.logger.info(f"  设备: {self.config.device}")
        self.logger.info("")

        # 模型参数信息
        model_size = self.model.get_model_size()
        self.logger.info("模型参数:")
        self.logger.info(f"  总参数量: {model_size['total_params']:,}")
        self.logger.info(f"  可训练参数量: {model_size['trainable_params']:,}")
        self.logger.info(f"  冻结参数量: {model_size['frozen_params']:,}")
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("")

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(self.train_loader, desc="训练中")

        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            user_ids = batch['user_id'].to(self.device)
            item_ids = batch['item_id'].to(self.device)
            ratings = batch['rating'].to(self.device)
            text_input_ids = batch['text_input_ids'].to(self.device)
            text_attention_mask = batch['text_attention_mask'].to(self.device)
            images = batch['image'].to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(user_ids, item_ids, text_input_ids,
                                   text_attention_mask, images)

            # 计算损失
            loss = self.criterion(predictions, ratings)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 更新参数
            self.optimizer.step()

            total_loss += loss.item()

            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
            })

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate_epoch(self):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="验证中"):
                # 移动数据到设备
                user_ids = batch['user_id'].to(self.device)
                item_ids = batch['item_id'].to(self.device)
                ratings = batch['rating'].to(self.device)
                text_input_ids = batch['text_input_ids'].to(self.device)
                text_attention_mask = batch['text_attention_mask'].to(self.device)
                images = batch['image'].to(self.device)

                # 前向传播
                predictions = self.model(user_ids, item_ids, text_input_ids,
                                       text_attention_mask, images)

                # 计算损失
                loss = self.criterion(predictions, ratings)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def train(self):
        """完整的训练过程"""
        self.logger.info("开始训练过程...")
        self.logger.info(f"训练批次数: {len(self.train_loader)}")
        self.logger.info(f"验证批次数: {len(self.val_loader)}")
        self.logger.info("")

        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            self.logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            self.logger.info("-" * 50)

            # 训练
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # 验证
            val_loss = self.validate_epoch()
            self.val_losses.append(val_loss)

            # 计算详细评估指标
            train_metrics, _ = evaluate_model(self.model, self.train_loader, self.device)
            val_metrics, _ = evaluate_model(self.model, self.val_loader, self.device)

            self.train_metrics_history.append(train_metrics)
            self.val_metrics_history.append(val_metrics)

            # 学习率调度
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']

            # 记录到TensorBoard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', new_lr, epoch)

            # 记录评估指标
            for metric_name, value in val_metrics.items():
                self.writer.add_scalar(f'Metrics/Val_{metric_name}', value, epoch)

            # 计算epoch用时
            epoch_time = time.time() - epoch_start_time

            # 记录详细的epoch结果
            self.log_epoch_results(epoch + 1, train_loss, val_loss, train_metrics,
                                 val_metrics, current_lr, new_lr, epoch_time)

            # 早停检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model('best_model.pth')
                self.logger.info("✓ 保存最佳模型")
            else:
                self.patience_counter += 1
                self.logger.info(f"⚠ 验证损失未改善 ({self.patience_counter}/{self.config.patience})")

                if self.patience_counter >= self.config.patience:
                    self.logger.info("🛑 早停触发，停止训练")
                    break

            self.logger.info("")

        training_time = time.time() - start_time

        # 记录训练完成信息
        self.log_training_summary(training_time)

        # 关闭TensorBoard
        self.writer.close()

        return self.train_losses, self.val_losses

    def log_epoch_results(self, epoch, train_loss, val_loss, train_metrics,
                         val_metrics, current_lr, new_lr, epoch_time):
        """记录epoch结果"""
        self.logger.info(f"训练损失: {train_loss:.6f}")
        self.logger.info(f"验证损失: {val_loss:.6f}")

        # 记录主要评估指标
        self.logger.info(f"验证MSE: {val_metrics.get('MSE', 0):.6f}")
        self.logger.info(f"验证MAE: {val_metrics.get('MAE', 0):.6f}")
        self.logger.info(f"验证RMSE: {val_metrics.get('RMSE', 0):.6f}")

        # 记录排序指标
        for k in [5, 10]:
            if f'Precision@{k}' in val_metrics:
                self.logger.info(f"验证Precision@{k}: {val_metrics[f'Precision@{k}']:.6f}")
                self.logger.info(f"验证Recall@{k}: {val_metrics[f'Recall@{k}']:.6f}")
                self.logger.info(f"验证NDCG@{k}: {val_metrics[f'NDCG@{k}']:.6f}")

        # 记录学习率变化
        if current_lr != new_lr:
            self.logger.info(f"学习率调整: {current_lr:.8f} → {new_lr:.8f}")
        else:
            self.logger.info(f"当前学习率: {current_lr:.8f}")

        # 记录epoch用时
        self.logger.info(f"Epoch用时: {epoch_time:.2f}秒")

    def log_training_summary(self, training_time):
        """记录训练总结"""
        self.logger.info("=" * 80)
        self.logger.info("训练完成总结")
        self.logger.info("=" * 80)
        self.logger.info(f"训练结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"总训练时间: {training_time:.2f}秒 ({training_time/60:.2f}分钟)")
        self.logger.info(f"完成轮次: {len(self.train_losses)}/{self.config.num_epochs}")

        if self.train_losses:
            self.logger.info(f"最终训练损失: {self.train_losses[-1]:.6f}")
            self.logger.info(f"最终验证损失: {self.val_losses[-1]:.6f}")
            self.logger.info(f"最佳验证损失: {self.best_val_loss:.6f}")

        # 记录最终评估指标
        if self.val_metrics_history:
            final_metrics = self.val_metrics_history[-1]
            self.logger.info("")
            self.logger.info("最终验证指标:")
            for metric_name, value in final_metrics.items():
                self.logger.info(f"  {metric_name}: {value:.6f}")

        self.logger.info("")
        self.logger.info(f"模型保存路径: {self.config.model_save_path}")
        self.logger.info(f"日志文件路径: {self.log_file_path}")
        self.logger.info("=" * 80)

    def save_model(self, filename):
        """保存模型"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        filepath = os.path.join(self.config.checkpoint_dir, filename)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics_history': self.train_metrics_history,
            'val_metrics_history': self.val_metrics_history
        }, filepath)

        print(f"模型已保存到: {filepath}")

    def load_model(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.train_metrics_history = checkpoint.get('train_metrics_history', [])
            self.val_metrics_history = checkpoint.get('val_metrics_history', [])

        print(f"模型已从 {filepath} 加载")

    def plot_training_curves(self, save_path=None):
        """绘制训练曲线"""
        if not self.train_losses or not self.val_losses:
            print("没有训练历史数据可绘制")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 损失曲线
        axes[0, 0].plot(self.train_losses, label='Training Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Validation Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # MSE曲线
        if self.val_metrics_history:
            val_mse = [m.get('MSE', 0) for m in self.val_metrics_history]
            axes[0, 1].plot(val_mse, label='Validation MSE', color='green')
            axes[0, 1].set_title('Validation MSE')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MSE')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

        # Precision@10曲线
        if self.val_metrics_history:
            val_precision = [m.get('Precision@10', 0) for m in self.val_metrics_history]
            axes[1, 0].plot(val_precision, label='Validation Precision@10', color='orange')
            axes[1, 0].set_title('Validation Precision@10')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision@10')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # NDCG@10曲线
        if self.val_metrics_history:
            val_ndcg = [m.get('NDCG@10', 0) for m in self.val_metrics_history]
            axes[1, 1].plot(val_ndcg, label='Validation NDCG@10', color='purple')
            axes[1, 1].set_title('Validation NDCG@10')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('NDCG@10')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练曲线已保存到: {save_path}")

        plt.close()  # 关闭图形以释放内存

        return fig
