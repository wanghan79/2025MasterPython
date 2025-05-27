from .base_evaluator import BaseEvaluator
from tele_fraud_detect import BertFraudDetector, BertAPI

class BertEvaluator(BaseEvaluator):
    def __init__(self, data_path, output_path, model_path):
        super().__init__(data_path, output_path)
        self.model_path = model_path
        self.label_map = {0: "非涉诈", 1: "涉诈"}
        
    def initialize_model(self):
        """初始化BERT模型"""
        self.logger.info('初始化BERT模型...')
        self.model_api = BertAPI(model_path=self.model_path, label_map=self.label_map)
        self.detector = BertFraudDetector(self.model_api)
        self.logger.info('BERT模型初始化完成')
        
    def process_item(self, item):
        """处理单条数据"""
        text = item["原始通话文本"]
        label = item["人工标注结果"]
        true_label = 1 if label in ["涉诈", "1", 1] else 0
        result = self.detector.detect(text)
        self.logger.info(result)
        pred = result.get("label", "非涉诈")
        pred_label = 0 if pred in ["非涉诈", "0", 0] else 1
        return true_label, pred_label, label, pred, text, result 