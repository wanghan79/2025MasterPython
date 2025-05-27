from .base_evaluator import BaseEvaluator
from tele_fraud_detect import FraudDetector, OpenAIAPI, FraudPrompt, FraudChoicePrompt

class OpenAIEvaluator(BaseEvaluator):
    def __init__(self, data_path, output_path, api_key, model, base_url, temperature=0.6, top_p=0.95, num_examples=0, prompt = "two_choice"):
        super().__init__(data_path, output_path)
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p
        if prompt == "two_choice":
            self.prompt_obj = FraudChoicePrompt()
        elif prompt == "CoT":
            self.prompt_obj = FraudPrompt()
        else:
            self.prompt_obj = FraudPrompt()
        self.prompt_obj.examples = self.prompt_obj.examples[:num_examples]

    def initialize_model(self):
        """初始化OpenAI API模型"""
        self.logger.info('初始化模型...')
        end_token = None
        if isinstance(self.prompt_obj, FraudChoicePrompt):
            end_token = self.prompt_obj.choice_end

        self.model_api = OpenAIAPI(
            api_key=self.api_key,
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            top_p=self.top_p,
            end_token=end_token
        )
        self.detector = FraudDetector(self.model_api, self.prompt_obj)
        self.logger.info('模型初始化完成')
        
    def process_item(self, item):
        """处理单条数据"""
        text = item["原始通话文本"]
        label = item["人工标注结果"]
        true_label = 1 if label in ["是", "1", 1] else 0
        
        raw_model_output = self.detector.detect(text) # 获取最原始的模型输出
        self.logger.info(f"MODEL_RAW_OUTPUT_CHECK: '{raw_model_output}' type: {type(raw_model_output)}")

        pred_text = "否" # 解析后的文本判断，默认为 "否"
        pred_label = 0    # 解析后的标签，默认为 0
        
        if isinstance(self.prompt_obj, FraudPrompt):
            self.logger.info(f"Processing with FraudPrompt. Raw model output: {raw_model_output}")
            
            if isinstance(raw_model_output, str):
                parsed_dict = None
                
                try:
                    json_str = raw_model_output
                    if json_str.startswith("```json"):
                        json_str = json_str[7:]
                    if json_str.endswith("```"):
                        json_str = json_str[:-3]
                    json_str = json_str.strip()
                    
                    start_index = json_str.find("{")
                    end_index = json_str.rfind("}")
                    if start_index != -1 and end_index != -1 and start_index < end_index:
                        json_str_to_parse = json_str[start_index : end_index+1]
                        import json
                        parsed_dict = json.loads(json_str_to_parse)
                        self.logger.info(f"成功解析模型输出为JSON格式: {parsed_dict}")
                except Exception as e:
                    self.logger.warning(f"使用 json.loads 解析失败: {str(e)}. 原始输出: {raw_model_output}. 尝试 ast.literal_eval.")

                if parsed_dict is None:
                    try:
                        import ast
                        evaluated_data = ast.literal_eval(raw_model_output.strip())
                        if isinstance(evaluated_data, dict):
                            parsed_dict = evaluated_data
                            self.logger.info(f"成功通过 ast.literal_eval 解析类字典结构: {parsed_dict}")
                        else:
                            self.logger.warning(f"ast.literal_eval 未返回字典类型，得到 {type(evaluated_data)}. 原始输出: {raw_model_output}")
                    except Exception as e:
                        self.logger.warning(f"使用 ast.literal_eval 解析也失败: {str(e)}. 原始输出: {raw_model_output}")

                if parsed_dict:
                    result_val_orig = parsed_dict.get("result")
                    classification_val_orig = parsed_dict.get("classification")

                    result_val_stripped = None
                    if isinstance(result_val_orig, str):
                        result_val_stripped = result_val_orig.strip()
                    elif result_val_orig is not None:
                        result_val_stripped = result_val_orig

                    classification_val_stripped = None
                    if isinstance(classification_val_orig, str):
                        classification_val_stripped = classification_val_orig.strip()
                    elif classification_val_orig is not None:
                        classification_val_stripped = classification_val_orig
                    
                    if result_val_stripped in ["是", "1", 1, True]:
                        pred_text = "是"; pred_label = 1
                    elif result_val_stripped in ["否", "0", 0, False]:
                        pred_text = "否"; pred_label = 0
                    elif classification_val_stripped in ["是", "1", 1, True]:
                        pred_text = "是"; pred_label = 1
                    elif classification_val_stripped in ["否", "0", 0, False]:
                        pred_text = "否"; pred_label = 0
                    else:
                        self.logger.warning(f"在解析的字典中未找到有效的 stripped 'result' ('{result_val_orig}') 或 'classification' ('{classification_val_orig}') 字段。 Parsed: {parsed_dict}. 尝试从原始字符串中正则匹配 'result' key.")
                        import re
                        pattern_yes = r"['\"]result['\"]\\s*:\\s*['\"]\\s*是\\s*['\"]"
                        pattern_no = r"['\"]result['\"]\\s*:\\s*['\"]\\s*否\\s*['\"]"
                        if re.search(pattern_yes, raw_model_output.lower()):
                            pred_text = "是"; pred_label = 1
                        elif re.search(pattern_no, raw_model_output.lower()):
                            pred_text = "否"; pred_label = 0
                            
                if pred_text == "否" and pred_label == 0:
                    self.logger.info(f"字典解析或特定键提取未能确定结果，回退到通用字符串级别判断。原始输出: {raw_model_output}")
                    s_lower = raw_model_output.strip().lower()
                    
                    if s_lower == "是":
                        pred_text = "是"; pred_label = 1
                    elif s_lower == "否":
                        pred_text = "否"; pred_label = 0
                    elif s_lower.startswith("是"):
                        pred_text = "是"; pred_label = 1
                    elif s_lower.startswith("否"):
                        pred_text = "否"; pred_label = 0
                    elif parsed_dict is None: # Only try this regex if dict parsing totally failed
                        import re
                        pattern_yes_fallback = r"['\"]result['\"]\\s*:\\s*['\"]\\s*是\\s*['\"]"
                        pattern_no_fallback = r"['\"]result['\"]\\s*:\\s*['\"]\\s*否\\s*['\"]"
                        if re.search(pattern_yes_fallback, s_lower):
                            pred_text = "是"; pred_label = 1
                        elif re.search(pattern_no_fallback, s_lower):
                            pred_text = "否"; pred_label = 0
                        elif "是" in raw_model_output and "否" not in raw_model_output:
                            pred_text = "是"; pred_label = 1
                        elif "否" in raw_model_output and "是" not in raw_model_output:
                            pred_text = "否"; pred_label = 0
                        else:
                             self.logger.warning(f"Fallback(dict parse failed): No clear '是'/'否' pattern. Output: {raw_model_output}")
                    elif "是" in raw_model_output and "否" not in raw_model_output:
                        pred_text = "是"; pred_label = 1
                    elif "否" in raw_model_output and "是" not in raw_model_output:
                        pred_text = "否"; pred_label = 0
                    else:
                        self.logger.warning(f"Fallback(all checks failed): No clear '是'/'否'. Output: {raw_model_output}")

            else: # raw_model_output 不是字符串
                 self.logger.warning(f"FraudPrompt: 期望模型输出为字符串，但得到 {type(raw_model_output)}: {raw_model_output}")

            return true_label, pred_label, label, pred_text, text, str(raw_model_output)

        elif isinstance(self.prompt_obj, FraudChoicePrompt):
            self.logger.info(f"Processing with FraudChoicePrompt. Raw model output: {raw_model_output}")
            
            if isinstance(raw_model_output, str):
                answer_str_cleaned = raw_model_output.strip()
                answer_str_lower = answer_str_cleaned.lower()
                
                # 1. 直接判断清理后的完整字符串是否为 "是" 或 "否"
                if answer_str_lower == "是":
                    pred_text = "是"
                    pred_label = 1
                elif answer_str_lower == "否":
                    pred_text = "否"
                    pred_label = 0
                else:
                    # 2. 尝试按冒号分割
                    parts = answer_str_cleaned.split(":", 1)
                    
                    if len(parts) > 1: # 包含冒号
                        value_after_colon = parts[1].strip().lower()
                        value_before_colon = parts[0].strip().lower()

                        if value_after_colon == "是":
                            pred_text = "是"
                            pred_label = 1
                        elif value_after_colon == "否":
                            pred_text = "否"
                            pred_label = 0
                        elif value_before_colon == "是": # 检查冒号前的内容
                            pred_text = "是"
                            pred_label = 1
                        elif value_before_colon == "否":
                            pred_text = "否"
                            pred_label = 0
                        else: 
                            self.logger.warning(f"FraudChoicePrompt: 冒号格式但冒号前后均无法解析为是/否: '{answer_str_cleaned}'. 尝试检查整体字符串是否以是/否开头.")
                            # 3. (作为冒号后处理的回退) 检查原始清理后的字符串是否以 "是" 或 "否" 开头 (忽略大小写)
                            if answer_str_lower.startswith("是"):
                                pred_text = "是"
                                pred_label = 1
                            elif answer_str_lower.startswith("否"):
                                pred_text = "否"
                                pred_label = 0
                            else:
                                self.logger.warning(f"FraudChoicePrompt: 冒号格式无法解析，且整体不以是/否开头: '{answer_str_cleaned}'. 默认为 '否'.")
                    else: # 不含冒号
                        # 3. (同样适用无冒号情况) 检查原始清理后的字符串是否以 "是" 或 "否" 开头 (忽略大小写)
                        if answer_str_lower.startswith("是"):
                            pred_text = "是"
                            pred_label = 1
                        elif answer_str_lower.startswith("否"):
                            pred_text = "否"
                            pred_label = 0
                        else:
                            self.logger.warning(f"FraudChoicePrompt: 无冒号且非标准 ('{answer_str_cleaned}'). 默认为 '否'.")
            else:
                self.logger.warning(f"FraudChoicePrompt: 期望模型输出为字符串，但得到 {type(raw_model_output)}: {raw_model_output}. 默认为 '否'.")

            return true_label, pred_label, label, pred_text, text, str(raw_model_output)
        
        raise ValueError(f"不支持的prompt类型: {type(self.prompt_obj)}")