import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_history(history):
    """绘制训练历史图表"""
    os.makedirs('results', exist_ok=True)

    plt.figure(figsize=(15, 10))

    # 绘制损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['test_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['test_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 绘制F1分数曲线
    plt.subplot(2, 2, 3)
    plt.plot(history['train_f1'], label='Training F1 Score')
    plt.plot(history['test_f1'], label='Validation F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/training_history.png')
    plt.show()

    print("Training history plots saved to results/training_history.png")