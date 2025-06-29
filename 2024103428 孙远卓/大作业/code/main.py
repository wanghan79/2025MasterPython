from train import train_model
from evaluate import evaluate_trained_model
from predict import SentimentPredictor
import config


def main():
    """主程序"""
    print("Starting Sentiment Analysis System")

    # 训练模型
    print("\n===== Training Phase =====")
    model, history = train_model()  # 修正这里：去掉下划线后的空格

    # 评估模型
    print("\n===== Evaluation Phase =====")
    evaluate_trained_model()

    # 预测示例
    print("\n===== Prediction Example =====")
    predictor = SentimentPredictor()

    sample_texts = [
        "This product is absolutely amazing! I love it.",
        "Terrible experience. Would not recommend to anyone.",
        "The item was okay, but delivery took too long.",
        "I'm satisfied with my purchase. It works as expected."
    ]

    for text in sample_texts:
        result = predictor.predict(text)
        print(f"\nText: {text}")
        print(f"Prediction: {result['sentiment']} (Confidence: {result['confidence']:.2f})")


if __name__ == "__main__":
    main()