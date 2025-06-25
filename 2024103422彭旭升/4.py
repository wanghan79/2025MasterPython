# DeepSeek-R1-8B Fine-tuning Project
# 完整的微调解决方案，支持自定义数据集和多种训练策略

import os
import json
import logging
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import pandas as pd
from typing import Dict, List, Optional, Union
import numpy as np
from dataclasses import dataclass, field
import yaml
from pathlib import Path
import wandb
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置类"""
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-8B"
    tokenizer_name: Optional[str] = None
    cache_dir: Optional[str] = "./cache"
    model_revision: str = "main"
    use_auth_token: Optional[str] = None
    torch_dtype: str = "float16"
    trust_remote_code: bool = True


@dataclass
class DataConfig:
    """数据配置类"""
    train_file: str = "train.jsonl"
    validation_file: Optional[str] = "validation.jsonl"
    test_file: Optional[str] = None
    max_seq_length: int = 2048
    preprocessing_num_workers: int = 4
    overwrite_cache: bool = False
    validation_split_percentage: int = 5
    text_column: str = "text"
    response_template: str = "### Response:"
    instruction_template: str = "### Instruction:"


@dataclass
class TrainingConfig:
    """训练配置类"""
    output_dir: str = "./deepseek-r1-finetuned"
    overwrite_output_dir: bool = True
    do_train: bool = True
    do_eval: bool = True
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    num_train_epochs: int = 3
    max_steps: int = -1
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    group_by_length: bool = True
    dataloader_pin_memory: bool = True
    report_to: str = "none"  # 可以设置为 "wandb" 如果要使用wandb
    run_name: Optional[str] = None
    seed: int = 42
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    deepspeed: Optional[str] = None


@dataclass
class LoraConfig_Custom:
    """LoRA配置类"""
    use_lora: bool = True
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class CustomDataset(Dataset):
    """自定义数据集类"""

    def __init__(self,
                 tokenizer,
                 file_path: str,
                 max_length: int = 2048,
                 text_column: str = "text",
                 instruction_template: str = "### Instruction:",
                 response_template: str = "### Response:"):

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.instruction_template = instruction_template
        self.response_template = response_template

        self.data = self._load_data(file_path)
        logger.info(f"Loaded {len(self.data)} examples from {file_path}")

    def _load_data(self, file_path: str) -> List[Dict]:
        """加载数据文件"""
        data = []
        if file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line.strip()))
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            data = df.to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        return data

    def _format_text(self, example: Dict) -> str:
        """格式化文本"""
        if "instruction" in example and "output" in example:
            # 指令-回答格式
            text = f"{self.instruction_template}\n{example['instruction']}\n\n{self.response_template}\n{example['output']}"
        elif "input" in example and "output" in example:
            # 输入-输出格式
            if example.get("instruction", ""):
                text = f"{self.instruction_template}\n{example['instruction']}\n\nInput: {example['input']}\n\n{self.response_template}\n{example['output']}"
            else:
                text = f"Input: {example['input']}\n\nOutput: {example['output']}"
        elif self.text_column in example:
            # 直接使用文本列
            text = example[self.text_column]
        else:
            # 尝试从其他字段构建
            text = str(example)

        return text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        text = self._format_text(example)

        # 分词
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()  # 对于语言建模，labels和input_ids相同
        }


class DataCollator:
    """数据整理器"""

    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        batch = {}

        # 获取最大长度
        max_len = min(
            max(len(ex["input_ids"]) for ex in examples),
            self.max_length
        )

        # 准备批次数据
        input_ids = []
        attention_mask = []
        labels = []

        for example in examples:
            # 截断或填充
            ids = example["input_ids"][:max_len]
            mask = example["attention_mask"][:max_len]
            label = example["labels"][:max_len]

            # 左填充
            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids = torch.cat([torch.full((pad_len,), self.tokenizer.pad_token_id), ids])
                mask = torch.cat([torch.zeros(pad_len), mask])
                label = torch.cat([torch.full((pad_len,), -100), label])

            input_ids.append(ids)
            attention_mask.append(mask)
            labels.append(label)

        batch["input_ids"] = torch.stack(input_ids)
        batch["attention_mask"] = torch.stack(attention_mask)
        batch["labels"] = torch.stack(labels)

        return batch


class ModelManager:
    """模型管理器"""

    def __init__(self, model_config: ModelConfig, lora_config: LoraConfig_Custom):
        self.model_config = model_config
        self.lora_config = lora_config
        self.tokenizer = None
        self.model = None

    def load_tokenizer(self):
        """加载分词器"""
        tokenizer_name = self.model_config.tokenizer_name or self.model_config.model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=self.model_config.cache_dir,
            revision=self.model_config.model_revision,
            use_auth_token=self.model_config.use_auth_token,
            trust_remote_code=self.model_config.trust_remote_code
        )

        # 设置特殊token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loaded tokenizer: {tokenizer_name}")
        return self.tokenizer

    def load_model(self):
        """加载模型"""
        torch_dtype = getattr(torch, self.model_config.torch_dtype)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            cache_dir=self.model_config.cache_dir,
            revision=self.model_config.model_revision,
            use_auth_token=self.model_config.use_auth_token,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=self.model_config.trust_remote_code
        )

        # 应用LoRA
        if self.lora_config.use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_config.r,
                lora_alpha=self.lora_config.lora_alpha,
                target_modules=self.lora_config.target_modules,
                lora_dropout=self.lora_config.lora_dropout,
                bias=self.lora_config.bias
            )

            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

        logger.info(f"Loaded model: {self.model_config.model_name}")
        return self.model


class Trainer_Custom(Trainer):
    """自定义训练器"""

    def compute_loss(self, model, inputs, return_outputs=False):
        """计算损失"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # 计算语言建模损失
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        else:
            loss = outputs.get("loss")

        return (loss, outputs) if return_outputs else loss


class FineTuner:
    """微调主类"""

    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.model_config = None
        self.data_config = None
        self.training_config = None
        self.lora_config = None

        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        self.model_config = ModelConfig(**config.get('model', {}))
        self.data_config = DataConfig(**config.get('data', {}))
        self.training_config = TrainingConfig(**config.get('training', {}))
        self.lora_config = LoraConfig_Custom(**config.get('lora', {}))

        logger.info(f"Loaded config from {config_path}")

    def setup_model_and_tokenizer(self):
        """设置模型和分词器"""
        model_manager = ModelManager(self.model_config, self.lora_config)
        self.tokenizer = model_manager.load_tokenizer()
        self.model = model_manager.load_model()

    def prepare_datasets(self):
        """准备数据集"""
        # 训练数据集
        if os.path.exists(self.data_config.train_file):
            self.train_dataset = CustomDataset(
                tokenizer=self.tokenizer,
                file_path=self.data_config.train_file,
                max_length=self.data_config.max_seq_length,
                text_column=self.data_config.text_column,
                instruction_template=self.data_config.instruction_template,
                response_template=self.data_config.response_template
            )
        else:
            raise FileNotFoundError(f"Training file not found: {self.data_config.train_file}")

        # 验证数据集
        if self.data_config.validation_file and os.path.exists(self.data_config.validation_file):
            self.eval_dataset = CustomDataset(
                tokenizer=self.tokenizer,
                file_path=self.data_config.validation_file,
                max_length=self.data_config.max_seq_length,
                text_column=self.data_config.text_column,
                instruction_template=self.data_config.instruction_template,
                response_template=self.data_config.response_template
            )
        else:
            # 从训练集中分割验证集
            train_size = len(self.train_dataset)
            eval_size = int(train_size * self.data_config.validation_split_percentage / 100)
            train_size = train_size - eval_size

            self.train_dataset, self.eval_dataset = torch.utils.data.random_split(
                self.train_dataset, [train_size, eval_size]
            )

        logger.info(f"Train dataset size: {len(self.train_dataset)}")
        logger.info(f"Eval dataset size: {len(self.eval_dataset)}")

    def create_trainer(self):
        """创建训练器"""
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            overwrite_output_dir=self.training_config.overwrite_output_dir,
            do_train=self.training_config.do_train,
            do_eval=self.training_config.do_eval,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            adam_beta1=self.training_config.adam_beta1,
            adam_beta2=self.training_config.adam_beta2,
            adam_epsilon=self.training_config.adam_epsilon,
            max_grad_norm=self.training_config.max_grad_norm,
            num_train_epochs=self.training_config.num_train_epochs,
            max_steps=self.training_config.max_steps,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            warmup_ratio=self.training_config.warmup_ratio,
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            eval_steps=self.training_config.eval_steps,
            evaluation_strategy=self.training_config.evaluation_strategy,
            save_strategy=self.training_config.save_strategy,
            save_total_limit=self.training_config.save_total_limit,
            load_best_model_at_end=self.training_config.load_best_model_at_end,
            metric_for_best_model=self.training_config.metric_for_best_model,
            greater_is_better=self.training_config.greater_is_better,
            group_by_length=self.training_config.group_by_length,
            dataloader_pin_memory=self.training_config.dataloader_pin_memory,
            report_to=self.training_config.report_to,
            run_name=self.training_config.run_name,
            seed=self.training_config.seed,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            deepspeed=self.training_config.deepspeed,
        )

        # 数据整理器
        data_collator = DataCollator(
            tokenizer=self.tokenizer,
            max_length=self.data_config.max_seq_length
        )

        # 创建训练器
        trainer = Trainer_Custom(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        return trainer

    def train(self):
        """开始训练"""
        logger.info("Starting training...")

        # 设置模型和分词器
        self.setup_model_and_tokenizer()

        # 准备数据集
        self.prepare_datasets()

        # 创建训练器
        trainer = self.create_trainer()

        # 检查是否有checkpoint
        last_checkpoint = None
        if os.path.isdir(self.training_config.output_dir):
            last_checkpoint = get_last_checkpoint(self.training_config.output_dir)

        # 开始训练
        if last_checkpoint is not None:
            logger.info(f"Resuming training from {last_checkpoint}")
            train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            train_result = trainer.train()

        # 保存模型
        trainer.save_model()
        trainer.save_state()

        # 保存训练结果
        output_train_file = os.path.join(self.training_config.output_dir, "train_results.txt")
        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # 评估
        if self.training_config.do_eval:
            logger.info("*** Evaluate ***")
            eval_results = trainer.evaluate()

            output_eval_file = os.path.join(self.training_config.output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(eval_results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

        logger.info("Training completed!")


def create_sample_config():
    """创建示例配置文件"""
    config = {
        'model': {
            'model_name': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-8B',
            'cache_dir': './cache',
            'torch_dtype': 'float16',
            'trust_remote_code': True
        },
        'data': {
            'train_file': 'train.jsonl',
            'validation_file': 'validation.jsonl',
            'max_seq_length': 2048,
            'validation_split_percentage': 5,
            'text_column': 'text',
            'instruction_template': '### Instruction:',
            'response_template': '### Response:'
        },
        'training': {
            'output_dir': './deepseek-r1-finetuned',
            'per_device_train_batch_size': 2,
            'per_device_eval_batch_size': 2,
            'gradient_accumulation_steps': 8,
            'learning_rate': 2e-4,
            'num_train_epochs': 3,
            'warmup_ratio': 0.03,
            'logging_steps': 10,
            'save_steps': 500,
            'eval_steps': 500,
            'save_total_limit': 3,
            'fp16': True,
            'gradient_checkpointing': True
        },
        'lora': {
            'use_lora': True,
            'r': 16,
            'lora_alpha': 32,
            'target_modules': ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            'lora_dropout': 0.1
        }
    }

    with open('config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print("Created sample config file: config.yaml")


def create_sample_data():
    """创建示例数据文件"""
    # 训练数据示例
    train_data = [
        {
            "instruction": "解释什么是机器学习",
            "output": "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。通过分析大量数据，机器学习算法可以识别模式并做出预测或决策。"
        },
        {
            "instruction": "什么是深度学习？",
            "output": "深度学习是机器学习的一个子集，使用人工神经网络来模拟人脑处理信息的方式。它在图像识别、自然语言处理和语音识别等领域表现出色。"
        },
        {
            "instruction": "解释什么是自然语言处理",
            "output": "自然语言处理（NLP）是计算机科学和人工智能的一个分支，专注于让计算机理解、解释和生成人类语言。它包括文本分析、机器翻译、情感分析等应用。"
        }
    ]

    # 保存训练数据
    with open('train.jsonl', 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 验证数据示例
    val_data = [
        {
            "instruction": "什么是强化学习？",
            "output": "强化学习是机器学习的一种类型，智能体通过与环境交互来学习最优行为策略。它通过试错和奖励反馈来改进决策过程。"
        }
    ]

    with open('validation.jsonl', 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("Created sample data files: train.jsonl, validation.jsonl")


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-R1-8B Fine-tuning")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--create-config", action="store_true", help="创建示例配置文件")
    parser.add_argument("--create-data", action="store_true", help="创建示例数据文件")

    args = parser.parse_args()

    if args.create_config:
        create_sample_config()
        return

    if args.create_data:
        create_sample_data()
        return

    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        logger.error(f"配置文件不存在: {args.config}")
        logger.info("使用 --create-config 创建示例配置文件")
        return

    # 开始微调
    fine_tuner = FineTuner(args.config)
    fine_tuner.train()


if __name__ == "__main__":
    main()