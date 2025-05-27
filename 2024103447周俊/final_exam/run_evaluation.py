import sys
sys.path.append('/data/tbsi/intern/zhouj/fraud-detect/tele_fraud_eval')
import logging
# from eval.bert_evaluator import BertEvaluator
# from eval.qwen3classify_evaluator import Qwen3ClassifyEvaluator
# from eval.openai_evaluator import OpenAIEvaluator
from .bert_evaluator import BertEvaluator
from .qwen3classify_evaluator import Qwen3ClassifyEvaluator
from .openai_evaluator import OpenAIEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # 训练集
    # data_path = "/data/tbsi/intern/zhouj/fraud-detect/tele_fraud_data/data/split-data/fraud-v3/train.json"
    # 测试数据路径
    data_path = "/data/tbsi/intern/zhouj/fraud-detect/tele_fraud_data/data/split-data/fraud-v3/test.json"
    # data_path  = "/data/tbsi/intern/dizs/fraud_detect/tele_fraud_data/data/jsondata/test_data.json"
    # logging.info(f"开始评估BERT模型")
    # # 1. BERT评估器示例
    # bert_evaluator = BertEvaluator(
    #     data_path=data_path,
    #     output_path="logs/bert-fraud-detect-v3-stable.json",
    #     model_path="/data/tbsi/intern/dizs/fraud_detect/tele_classify/logs/bert_model/bert-fraud-detect-v3-stable/checkpoint-1074"
    # )
    # bert_evaluator.evaluate()
    # logging.info(f"BERT模型评估完成")

    # # 2. Qwen3Classify评估器示例
    # qwen3_evaluator = Qwen3ClassifyEvaluator(
    #     data_path=data_path,
    #     output_path="logs/eval_qwen3classify_results.json",
    #     model_path="/data/tbsi/intern/dizs/fraud_detect/LLaMA-Factory/output/qwen3_fraud_detect-v2",
    #     classifier_path="/data/tbsi/intern/dizs/fraud_detect/tele_classify/logs/qwen3_fraud_detect-v2/best_classifier.pt"
    # )
    # qwen3_evaluator.evaluate()
    
    # 3. OpenAI API评估器示例
    # openai_evaluator = OpenAIEvaluator(
    #     data_path=data_path,
    #     output_path="logs/qwen2.5-72b-choice-0-shot-Test/eval_results.json",
    #     api_key="sk-MZH2vaMPWT5HKQNcATFM",
    #     model="Qwen/Qwen2.5-72B-Instruct",
    #     base_url="http://10.97.236.70:4000/v1",
    #     temperature=0.6,
    #     top_p=0.95,
    #     num_examples=0,
    #     prompt="two_choice"
    # )
    openai_evaluator = OpenAIEvaluator(
        data_path=data_path,
        output_path="logs/qwen3-4B-v3-5-23-Full-test/eval_results.json",
        # raw_response_log_path="logs/qwen3-4B-v3-5-23-demo-test/all_raw_model_responses.txt",
        api_key="sk-MZH2vaMPWT5HKQNcATFM",
        model="Qwen3-4B-v3",
        base_url="http://localhost:8322/v1",
        temperature=0.6,
        top_p=0.95,
        num_examples=2,
        # num_examples=0,
        prompt="two_choice"  # "two_choice" or "one_choice",
    )
    openai_evaluator.evaluate()

if __name__ == "__main__":
    main() 