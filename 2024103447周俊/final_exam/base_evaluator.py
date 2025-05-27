# import json
# import logging
# import concurrent.futures
# from tqdm import tqdm
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from abc import ABC, abstractmethod

# class BaseEvaluator(ABC):
#     def __init__(self, data_path, output_path):
#         self.data_path = data_path
#         self.output_path = output_path
#         self.logger = logging.getLogger(self.__class__.__name__)
#         self.logger.setLevel(logging.INFO)
        
#         # 初始化评估指标
#         self.y_true = []
#         self.y_pred = []
#         self.results = []
        
#     def load_data(self):
#         """加载测试数据"""
#         self.logger.info('开始加载数据...')
#         with open(self.data_path, "r", encoding="utf-8") as f:
#             self.data = json.load(f)
#         self.logger.info(f'数据加载完成，共{len(self.data)}条')
        
#     @abstractmethod
#     def initialize_model(self):
#         """初始化模型，由子类实现"""
#         pass
        
#     @abstractmethod
#     def process_item(self, item):
#         """处理单条数据，由子类实现"""
#         pass
        
#     def evaluate(self):
#         """执行评估流程"""
#         # 1. 加载数据
#         self.load_data()
        
#         # 2. 初始化模型
#         self.initialize_model()
        
#         # 3. 处理数据并收集结果
#         with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
#             futures = [executor.submit(self.process_item, item) for item in self.data]
#             for idx, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures))):
#                 true_label, pred_label, label, pred, text, result = future.result()
#                 self.y_true.append(true_label)
#                 self.y_pred.append(pred_label)
#                 self.logger.info(f'第{idx+1}条 | 标注: {label} | 预测: {pred} | 文本: {text[:30]}...')
                
#                 # 保存结果
#                 self.results.append({
#                     "question": text,
#                     "model_output": pred,
#                     "label": true_label,
#                     "predicted_label": pred_label,
#                     "model_raw_output": result
#                 })
                
#         # 4. 保存结果
#         self.save_results()
        
#         # 5. 计算并输出指标
#         self.calculate_metrics()
        
#     def save_results(self):
#         """保存评估结果"""
#         with open(self.output_path, "w", encoding="utf-8") as f:
#             json.dump(self.results, f, ensure_ascii=False, indent=2)
            
#     def calculate_metrics(self):
#         """计算评估指标"""
#         acc = accuracy_score(self.y_true, self.y_pred)
#         recall = recall_score(self.y_true, self.y_pred)
#         precision = precision_score(self.y_true, self.y_pred)
#         f1 = f1_score(self.y_true, self.y_pred)
        
#         self.logger.info(f"准确率(Accuracy): {acc:.4f}")
#         self.logger.info(f"召回率(Recall): {recall:.4f}")
#         self.logger.info(f"精确率(Precision): {precision:.4f}")
#         self.logger.info(f"F1分数: {f1:.4f}") 

import json
import logging
import os
import concurrent.futures
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        # # 原始响应路径
        # self.raw_response_log_path = raw_response_log_path
        # # 初始化原始响应
        # self.raw_responses_log_data = []
        # 初始化评估指标
        self.y_true = []
        self.y_pred = []
        self.results = []
        
    def load_data(self):
        """加载测试数据"""
        self.logger.info('开始加载数据...')
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.logger.info(f'数据加载完成，共{len(self.data)}条')
        
    @abstractmethod
    def initialize_model(self):
        """初始化模型，由子类实现"""
        pass
        
    @abstractmethod
    def process_item(self, item):
        """处理单条数据，由子类实现"""
        pass
        
    def evaluate(self):
        """执行评估流程"""
        # 1. 加载数据
        self.load_data()
        
        # 2. 初始化模型
        self.initialize_model()
        
        # 3. 处理数据并收集结果
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.process_item, item) for item in self.data]
            for idx, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures))):
                true_label, pred_label, label, pred, text, result = future.result()
                self.y_true.append(true_label)
                self.y_pred.append(pred_label)
                self.logger.info(f'第{idx+1}条 | 标注: {label} | 预测: {pred} | 文本: {text[:30]}...')
                
                # 保存结果
                self.results.append({
                    "question": text,
                    "model_output": pred,
                    "label": true_label,
                    "predicted_label": pred_label,
                    "model_raw_output": result
                })
                # # 假设存在一个原始响应的保存路径就用
                # if self.raw_response_log_path:
                #     self.raw_responses_log_data.append({
                #         "item_index": idx + 1,
                #         "input_text_snippet": text[:100] + "...", # Log more snippet
                #         "true_label": label,
                #         "predicted_output": pred,
                #         "raw_response": result
                #     })
        # 4. 保存结果
        self.save_results()
        # # 保存原始响应
        # if self.raw_response_log_path:
        #     self.save_raw_responses_log()
        
        # 5. 计算并输出指标
        self.calculate_metrics()
        
    def save_results(self):
        """保存评估结果"""
        # Ensure the output directory exists
        output_dir = os.path.dirname(self.output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            self.logger.info(f"创建目录: {output_dir}")
            
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
            
    def calculate_metrics(self):
        """计算评估指标"""
        acc = accuracy_score(self.y_true, self.y_pred)
        recall = recall_score(self.y_true, self.y_pred)
        precision = precision_score(self.y_true, self.y_pred)
        f1 = f1_score(self.y_true, self.y_pred)
        
        self.logger.info(f"准确率(Accuracy): {acc:.4f}")
        self.logger.info(f"召回率(Recall): {recall:.4f}")
        self.logger.info(f"精确率(Precision): {precision:.4f}")
        self.logger.info(f"F1分数: {f1:.4f}") 
    
    # def save_raw_responses_log(self):
    #     """保存原始模型响应日志"""
    #     if not self.raw_response_log_path or not self.raw_responses_log_data:
    #         return
            
    #     self.logger.info(f"开始保存原始模型响应到: {self.raw_response_log_path}")
    #     output_dir = os.path.dirname(self.raw_response_log_path)
    #     if output_dir and not os.path.exists(output_dir): # Check if output_dir is not empty
    #         os.makedirs(output_dir)
    #         self.logger.info(f"创建目录: {output_dir}")
            
    #     try:
    #         with open(self.raw_response_log_path, "w", encoding="utf-8") as f:
    #             for log_item in self.raw_responses_log_data:
    #                 f.write(f"--- Item {log_item['item_index']} ---\n")
    #                 f.write(f"Input Text (first 100 chars): {log_item['input_text_snippet']}\n")
    #                 f.write(f"True Label: {log_item['true_label']}\n")
    #                 f.write(f"Predicted Output: {log_item['predicted_output']}\n")
    #                 f.write("Raw Model Response:\n")
    #                 # Handle if raw_response is a dict/list (like from OpenAI) or a simple string
    #                 if isinstance(log_item['raw_response'], (dict, list)):
    #                     f.write(json.dumps(log_item['raw_response'], ensure_ascii=False, indent=2))
    #                 else:
    #                     f.write(str(log_item['raw_response']))
    #                 f.write("\n\n")
    #         self.logger.info(f"原始模型响应已保存到: {self.raw_response_log_path}")
    #     except IOError as e:
    #         self.logger.error(f"无法写入原始模型响应日志文件 '{self.raw_response_log_path}': {e}") 