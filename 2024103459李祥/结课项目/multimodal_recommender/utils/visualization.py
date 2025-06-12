"""
可视化工具模块
包含训练曲线绘制、模型架构图等可视化功能
"""

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
import os
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches


def plot_training_curves(train_losses, val_losses, train_metrics=None, val_metrics=None,
                        save_path=None, figsize=(15, 10)):
    """
    绘制训练曲线

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_metrics: 训练指标历史
        val_metrics: 验证指标历史
        save_path: 保存路径
        figsize: 图像大小
    """
    try:
        plt.style.use('seaborn-v0_8')
    except:
        try:
            plt.style.use('seaborn')
        except:
            pass  # 使用默认样式

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 损失曲线
    epochs = range(1, len(train_losses) + 1)
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    # MSE曲线
    if val_metrics:
        val_mse = [m.get('MSE', 0) for m in val_metrics]
        axes[0, 1].plot(epochs, val_mse, 'g-', label='Validation MSE', linewidth=2)
        axes[0, 1].set_title('Validation MSE', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('MSE', fontsize=12)
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)

    # Precision@10曲线
    if val_metrics:
        val_precision = [m.get('Precision@10', 0) for m in val_metrics]
        axes[1, 0].plot(epochs, val_precision, 'orange', label='Validation Precision@10', linewidth=2)
        axes[1, 0].set_title('Validation Precision@10', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Precision@10', fontsize=12)
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)

    # NDCG@10曲线
    if val_metrics:
        val_ndcg = [m.get('NDCG@10', 0) for m in val_metrics]
        axes[1, 1].plot(epochs, val_ndcg, 'purple', label='Validation NDCG@10', linewidth=2)
        axes[1, 1].set_title('Validation NDCG@10', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('NDCG@10', fontsize=12)
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")

    plt.close()  # 关闭图形以释放内存
    return fig


def plot_model_architecture(save_path=None, figsize=(16, 12)):
    """
    绘制模型架构图

    Args:
        save_path: 保存路径
        figsize: 图像大小
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # 定义颜色
    colors = {
        'input': '#E8F4FD',
        'embedding': '#B3E5FC',
        'feature': '#81C784',
        'fusion': '#FFB74D',
        'transformer': '#F06292',
        'output': '#CE93D8'
    }

    # 输入层
    input_boxes = [
        {'xy': (0.5, 10), 'width': 1.5, 'height': 0.8, 'text': 'User ID\nItem ID', 'color': colors['input']},
        {'xy': (2.5, 10), 'width': 1.5, 'height': 0.8, 'text': 'Text\nDescription', 'color': colors['input']},
        {'xy': (4.5, 10), 'width': 1.5, 'height': 0.8, 'text': 'Item\nImage', 'color': colors['input']}
    ]

    for box in input_boxes:
        rect = FancyBboxPatch(
            box['xy'], box['width'], box['height'],
            boxstyle="round,pad=0.1",
            facecolor=box['color'],
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(box['xy'][0] + box['width']/2, box['xy'][1] + box['height']/2,
                box['text'], ha='center', va='center', fontsize=10, fontweight='bold')

    # 嵌入层
    embedding_boxes = [
        {'xy': (0.5, 8.5), 'width': 1.5, 'height': 0.8, 'text': 'User & Item\nEmbedding', 'color': colors['embedding']},
        {'xy': (2.5, 8.5), 'width': 1.5, 'height': 0.8, 'text': 'Text\nEncoder', 'color': colors['feature']},
        {'xy': (4.5, 8.5), 'width': 1.5, 'height': 0.8, 'text': 'Image\nEncoder', 'color': colors['feature']}
    ]

    for box in embedding_boxes:
        rect = FancyBboxPatch(
            box['xy'], box['width'], box['height'],
            boxstyle="round,pad=0.1",
            facecolor=box['color'],
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(box['xy'][0] + box['width']/2, box['xy'][1] + box['height']/2,
                box['text'], ha='center', va='center', fontsize=10, fontweight='bold')

    # 特征投影层
    proj_boxes = [
        {'xy': (0.5, 7), 'width': 1.5, 'height': 0.6, 'text': 'Feature\nProjection', 'color': colors['embedding']},
        {'xy': (2.5, 7), 'width': 1.5, 'height': 0.6, 'text': 'Feature\nProjection', 'color': colors['embedding']},
        {'xy': (4.5, 7), 'width': 1.5, 'height': 0.6, 'text': 'Feature\nProjection', 'color': colors['embedding']}
    ]

    for box in proj_boxes:
        rect = FancyBboxPatch(
            box['xy'], box['width'], box['height'],
            boxstyle="round,pad=0.1",
            facecolor=box['color'],
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(box['xy'][0] + box['width']/2, box['xy'][1] + box['height']/2,
                box['text'], ha='center', va='center', fontsize=9, fontweight='bold')

    # 多模态融合层
    fusion_box = FancyBboxPatch(
        (1.5, 5.5), 3, 1,
        boxstyle="round,pad=0.1",
        facecolor=colors['fusion'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(fusion_box)
    ax.text(3, 6, 'Multi-Modal Fusion\n(Multi-Head Attention)', ha='center', va='center',
            fontsize=12, fontweight='bold')

    # Transformer编码器
    transformer_box = FancyBboxPatch(
        (1.5, 4), 3, 1,
        boxstyle="round,pad=0.1",
        facecolor=colors['transformer'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(transformer_box)
    ax.text(3, 4.5, 'Transformer Encoder\n(Self-Attention)', ha='center', va='center',
            fontsize=12, fontweight='bold')

    # 预测层
    output_box = FancyBboxPatch(
        (1.5, 2.5), 3, 1,
        boxstyle="round,pad=0.1",
        facecolor=colors['output'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(output_box)
    ax.text(3, 3, 'Prediction Layer\n(Fully Connected)', ha='center', va='center',
            fontsize=12, fontweight='bold')

    # 输出
    final_output = FancyBboxPatch(
        (2.25, 1), 1.5, 0.8,
        boxstyle="round,pad=0.1",
        facecolor=colors['output'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(final_output)
    ax.text(3, 1.4, 'Rating\nPrediction', ha='center', va='center',
            fontsize=11, fontweight='bold')

    # 绘制箭头
    arrow_props = dict(arrowstyle='->', lw=2, color='black')

    # 输入到嵌入层的箭头
    for i in range(3):
        ax.annotate('', xy=(0.5 + i*2 + 0.75, 8.5), xytext=(0.5 + i*2 + 0.75, 10),
                   arrowprops=arrow_props)

    # 嵌入层到投影层的箭头
    for i in range(3):
        ax.annotate('', xy=(0.5 + i*2 + 0.75, 7), xytext=(0.5 + i*2 + 0.75, 8.5),
                   arrowprops=arrow_props)

    # 投影层到融合层的箭头
    for i in range(3):
        ax.annotate('', xy=(3, 6.5), xytext=(0.5 + i*2 + 0.75, 7),
                   arrowprops=arrow_props)

    # 融合层到Transformer的箭头
    ax.annotate('', xy=(3, 5), xytext=(3, 5.5), arrowprops=arrow_props)

    # Transformer到预测层的箭头
    ax.annotate('', xy=(3, 3.5), xytext=(3, 4), arrowprops=arrow_props)

    # 预测层到输出的箭头
    ax.annotate('', xy=(3, 1.8), xytext=(3, 2.5), arrowprops=arrow_props)

    # 添加标题
    ax.text(5, 11.5, 'Multi-Modal Transformer Recommender System Architecture',
            fontsize=16, fontweight='bold', ha='center')

    # 添加图例
    legend_elements = [
        mpatches.Patch(color=colors['input'], label='Input Layer'),
        mpatches.Patch(color=colors['embedding'], label='Embedding/Projection Layer'),
        mpatches.Patch(color=colors['feature'], label='Feature Extraction Layer'),
        mpatches.Patch(color=colors['fusion'], label='Multi-Modal Fusion Layer'),
        mpatches.Patch(color=colors['transformer'], label='Transformer Layer'),
        mpatches.Patch(color=colors['output'], label='Output Layer')
    ]

    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model architecture saved to: {save_path}")

    plt.close()  # 关闭图形以释放内存
    return fig


def plot_metrics_comparison(metrics_dict, save_path=None, figsize=(12, 8)):
    """
    绘制不同模型的指标对比图

    Args:
        metrics_dict: 指标字典，格式为 {'model_name': metrics}
        save_path: 保存路径
        figsize: 图像大小
    """
    if not metrics_dict:
        print("没有指标数据可绘制")
        return

    # 提取指标名称
    all_metrics = set()
    for metrics in metrics_dict.values():
        all_metrics.update(metrics.keys())

    # 过滤出需要的指标
    target_metrics = ['MSE', 'MAE', 'Precision@5', 'Recall@5', 'NDCG@5', 'Hit_Ratio@5']
    available_metrics = [m for m in target_metrics if m in all_metrics]

    if not available_metrics:
        print("没有可用的指标数据")
        return

    # 准备数据
    models = list(metrics_dict.keys())
    metric_values = {metric: [metrics_dict[model].get(metric, 0) for model in models]
                    for metric in available_metrics}

    # 创建子图
    n_metrics = len(available_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # 绘制每个指标
    for i, metric in enumerate(available_metrics):
        row = i // n_cols
        col = i % n_cols

        ax = axes[row, col]
        bars = ax.bar(models, metric_values[metric], alpha=0.7)
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=10)

        # 添加数值标签
        for bar, value in zip(bars, metric_values[metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=9)

        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for i in range(n_metrics, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"指标对比图已保存到: {save_path}")

    plt.show()
    return fig
