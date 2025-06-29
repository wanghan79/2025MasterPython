import os
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
import math
from typing import Tuple, List, Dict, Set, Optional, Union
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer
from tqdm import tqdm
import sys
import logging
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import random

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tkbc_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TKBC")

# 数据路径配置
DATA_PATH = Path("./datasets")
MODEL_PATH = Path("./models")

@dataclass
class DatasetConfig:
    """数据集配置类"""
    name: str
    path: str
    train_file: str = "train"
    valid_file: str = "valid"
    test_file: str = "test"
    delimiter: str = "\t"
    has_time: bool = True
    time_format: str = "YYYY-MM-DD"


class TemporalKnowledgeGraphDataset(Dataset):
    """时序知识图谱数据集类"""
    def __init__(self, data: np.ndarray):
        self.data = torch.LongTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class BasePreprocessor:
    """数据集预处理基类"""
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.entities = set()
        self.relations = set()
        self.timestamps = set()
        self.entities_to_id = {}
        self.relations_to_id = {}
        self.timestamps_to_id = {}
        self.n_entities = 0
        self.n_relations = 0
        self.n_timestamps = 0
        self.save_dir = DATA_PATH / config.name
        self.files = [config.train_file, config.valid_file, config.test_file]
    
    def _read_file(self, file_path: str) -> List[str]:
        """读取文件内容并处理异常"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logger.error(f"文件不存在: {file_path}")
            return []
        except UnicodeDecodeError:
            logger.error(f"文件编码错误: {file_path}, 尝试使用ISO-8859-1编码")
            with open(file_path, 'r', encoding='ISO-8859-1') as f:
                return [line.strip() for line in f if line.strip()]
    
    def _process_line(self, line: str) -> Tuple:
        """处理数据行，需由子类实现"""
        raise NotImplementedError("子类必须实现_process_line方法")
    
    def collect_entities_relations_timestamps(self) -> None:
        """收集实体、关系和时间戳"""
        logger.info(f"开始收集{self.config.name}数据集的实体、关系和时间戳...")
        for file in self.files:
            file_path = os.path.join(self.config.path, file)
            lines = self._read_file(file_path)
            for line in lines:
                elements = self._process_line(line)
                self.entities.add(elements[0])
                self.entities.add(elements[2])
                self.relations.add(elements[1])
                if self.config.has_time:
                    self.timestamps.add(elements[3])
    
    def create_mappings(self) -> None:
        """创建实体、关系和时间戳的映射"""
        logger.info(f"创建{self.config.name}数据集的映射关系...")
        self.entities_to_id = {x: i for i, x in enumerate(sorted(self.entities))}
        self.relations_to_id = {x: i for i, x in enumerate(sorted(self.relations))}
        if self.config.has_time:
            self.timestamps_to_id = {x: i for i, x in enumerate(sorted(self.timestamps))}
        
        self.n_entities = len(self.entities)
        self.n_relations = len(self.relations)
        self.n_timestamps = len(self.timestamps)
        
        logger.info(f"{self.config.name}数据集统计: {self.n_entities}实体, {self.n_relations}关系, {self.n_timestamps}时间戳")
    
    def save_mappings(self) -> None:
        """保存映射关系到文件"""
        logger.info(f"保存{self.config.name}数据集的映射关系...")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        mapping_files = [
            (self.entities_to_id, "ent_id"),
            (self.relations_to_id, "rel_id")
        ]
        if self.config.has_time:
            mapping_files.append((self.timestamps_to_id, "ts_id"))
        
        for mapping, file_name in mapping_files:
            file_path = self.save_dir / file_name
            with open(file_path, 'w', encoding='utf-8') as f:
                for key, value in mapping.items():
                    f.write(f"{key}\t{value}\n")
    
    def process_examples(self) -> Dict[str, np.ndarray]:
        """处理数据集示例并转换为ID"""
        logger.info(f"处理{self.config.name}数据集的示例...")
        examples = {file: [] for file in self.files}
        
        for file in self.files:
            file_path = os.path.join(self.config.path, file)
            lines = self._read_file(file_path)
            for line in lines:
                elements = self._process_line(line)
                try:
                    if self.config.has_time:
                        example = [
                            self.entities_to_id[elements[0]],
                            self.relations_to_id[elements[1]],
                            self.entities_to_id[elements[2]],
                            self.timestamps_to_id[elements[3]]
                        ]
                    else:
                        example = [
                            self.entities_to_id[elements[0]],
                            self.relations_to_id[elements[1]],
                            self.entities_to_id[elements[2]]
                        ]
                    examples[file].append(example)
                except KeyError as e:
                    logger.warning(f"跳过包含未知元素的行: {line}, 错误: {e}")
            
            if examples[file]:
                examples[file] = np.array(examples[file], dtype='uint64')
                with open(self.save_dir / f"{file}.pickle", 'wb') as f:
                    pickle.dump(examples[file], f)
            else:
                logger.warning(f"文件{file}处理后没有有效示例")
                examples[file] = np.empty((0, 4), dtype='uint64') if self.config.has_time else np.empty((0, 3), dtype='uint64')
        
        return examples
    
    def create_filtering_lists(self, examples: Dict[str, np.ndarray]) -> None:
        """创建过滤列表"""
        logger.info(f"为{self.config.name}数据集创建过滤列表...")
        to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
        
        for file, data in examples.items():
            for lhs, rel, rhs, ts in data:
                to_skip['lhs'][(rhs, rel + self.n_relations, ts)].add(lhs)
                to_skip['rhs'][(lhs, rel, ts)].add(rhs)
        
        to_skip_final = {
            'lhs': {k: sorted(list(v)) for k, v in to_skip['lhs'].items()},
            'rhs': {k: sorted(list(v)) for k, v in to_skip['rhs'].items()}
        }
        
        with open(self.save_dir / "to_skip.pickle", 'wb') as f:
            pickle.dump(to_skip_final, f)
    
    def calculate_probabilities(self, train_examples: np.ndarray) -> None:
        """计算实体出现概率"""
        logger.info(f"计算{self.config.name}数据集的实体出现概率...")
        counters = {
            'lhs': np.zeros(self.n_entities, dtype=np.float32),
            'rhs': np.zeros(self.n_entities, dtype=np.float32),
            'both': np.zeros(self.n_entities, dtype=np.float32)
        }
        
        for lhs, rel, rhs, _ in train_examples:
            counters['lhs'][lhs] += 1
            counters['rhs'][rhs] += 1
            counters['both'][lhs] += 1
            counters['both'][rhs] += 1
        
        # 归一化
        for key in counters:
            total = np.sum(counters[key])
            if total > 0:
                counters[key] = counters[key] / total
        
        with open(self.save_dir / "probas.pickle", 'wb') as f:
            pickle.dump(counters, f)
    
    def preprocess(self) -> None:
        """执行完整的预处理流程"""
        logger.info(f"开始预处理{self.config.name}数据集...")
        self.collect_entities_relations_timestamps()
        self.create_mappings()
        self.save_mappings()
        examples = self.process_examples()
        self.create_filtering_lists(examples)
        if self.config.train_file in examples and examples[self.config.train_file].size > 0:
            self.calculate_probabilities(examples[self.config.train_file])
        logger.info(f"{self.config.name}数据集预处理完成")


class ICEWSpreprocessor(BasePreprocessor):
    """ICEWS数据集预处理类"""
    def _process_line(self, line: str) -> Tuple[str, str, str, str]:
        """处理ICEWS数据行"""
        parts = line.split(self.config.delimiter)
        if len(parts) != 4:
            logger.warning(f"数据行格式错误: {line}, 期望4列，实际{len(parts)}列")
            return ("", "", "", "")
        return (parts[0], parts[1], parts[2], parts[3])


class Wiki12KPreprocessor(BasePreprocessor):
    """wiki12K数据集预处理类"""
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.all_ts = []
    
    def _parse_time(self, time_str: str) -> Tuple[int, int, int]:
        """解析时间字符串"""
        if time_str == "####":
            return (-math.inf, 0, 0)
        parts = time_str.strip().split('-')
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 0
        day = int(parts[2]) if len(parts) > 2 else 0
        return (year, month, day)
    
    def _process_line(self, line: str) -> Tuple[str, str, str, str, str]:
        """处理wiki12K数据行"""
        parts = line.split(self.config.delimiter)
        if len(parts) != 5:
            logger.warning(f"数据行格式错误: {line}, 期望5列，实际{len(parts)}列")
            return ("", "", "", "", "")
        return (parts[0], parts[1], parts[2], parts[3], parts[4])
    
    def create_mappings(self) -> None:
        """创建映射关系，处理时间戳特殊逻辑"""
        super().create_mappings()
        if self.config.has_time:
            self.all_ts = sorted(self.timestamps)[1:-1]  # 排除首尾特殊时间
            self.timestamps_to_id = {x: i for i, x in enumerate(self.all_ts)}
            self.n_timestamps = len(self.all_ts)
            logger.info(f"处理后时间戳数量: {self.n_timestamps}")
    
    def process_examples(self) -> Dict[str, np.ndarray]:
        """处理示例并处理时间间隔"""
        logger.info(f"处理{self.config.name}数据集的时间间隔示例...")
        examples = {file: [] for file in self.files}
        event_list = {'all': []}
        
        for file in self.files:
            file_path = os.path.join(self.config.path, file)
            lines = self._read_file(file_path)
            stats = {
                'ignored': 0,
                'total': 0,
                'full_intervals': 0,
                'half_intervals': 0,
                'point': 0
            }
            
            for line in lines:
                stats['total'] += 1
                lhs, rel, rhs, begin, end = self._process_line(line)
                
                begin_t = self._parse_time(begin)
                end_t = self._parse_time(end)
                
                # 处理特殊时间范围
                if begin_t[0] == -math.inf:
                    begin = self.all_ts[0]
                    if end_t[0] != math.inf:
                        stats['half_intervals'] += 1
                if end_t[0] == math.inf:
                    end = self.all_ts[-1]
                    if begin_t[0] != -math.inf:
                        stats['half_intervals'] += 1
                
                # 统计时间类型
                if begin_t[0] > -math.inf and end_t[0] < math.inf:
                    if begin_t[0] == end_t[0]:
                        stats['point'] += 1
                    else:
                        stats['full_intervals'] += 1
                
                # 转换为ID
                try:
                    begin_id = self.timestamps_to_id[begin]
                    end_id = self.timestamps_to_id[end]
                    lhs_id = self.entities_to_id[lhs]
                    rel_id = self.relations_to_id[rel]
                    rhs_id = self.entities_to_id[rhs]
                except KeyError as e:
                    stats['ignored'] += 1
                    logger.warning(f"元素ID映射错误: {e}")
                    continue
                
                # 处理时间顺序
                if begin_id > end_id:
                    stats['ignored'] += 1
                    continue
                
                # 添加事件列表
                event_list['all'].append((begin_id, -1, (lhs_id, rel_id, rhs_id)))
                event_list['all'].append((end_id, +1, (lhs_id, rel_id, rhs_id)))
                
                # 添加到示例
                examples[file].append([lhs_id, rel_id, rhs_id, begin_id, end_id])
            
            if examples[file]:
                examples[file] = np.array(examples[file], dtype='uint64')
                with open(self.save_dir / f"{file}.pickle", 'wb') as f:
                    pickle.dump(examples[file], f)
                logger.info(f"{file} - 忽略事件: {stats['ignored']}, 总事件: {stats['total']}, "
                            f"完整间隔: {stats['full_intervals']}, 半间隔: {stats['half_intervals']}, "
                            f"点事件: {stats['point']}")
            else:
                logger.warning(f"文件{file}处理后没有有效示例")
                examples[file] = np.empty((0, 5), dtype='uint64')
        
        # 保存事件列表
        for key, events in event_list.items():
            if events:
                with open(self.save_dir / f"event_list_{key}.pickle", 'wb') as f:
                    logger.info(f"保存{len(events)}个事件到event_list_{key}.pickle")
                    pickle.dump(sorted(events), f)
        
        # 计算时间差
        self._calculate_time_differences()
        
        return examples
    
    def _calculate_time_differences(self) -> None:
        """计算时间戳之间的差异"""
        logger.info(f"计算{self.config.name}数据集的时间差...")
        if not self.all_ts or len(self.all_ts) < 2:
            logger.warning("时间戳数量不足，无法计算时间差")
            return
        
        try:
            ts_to_int = [ts[0] for ts in self.all_ts]  # 仅使用年份简化计算
            ts = np.array(ts_to_int, dtype='float32')
            diffs = ts[1:] - ts[:-1]  # 计算相邻时间差
            with open(self.save_dir / "ts_diffs.pickle", 'wb') as f:
                pickle.dump(diffs, f)
            logger.info(f"时间差计算完成，保存到ts_diffs.pickle")
        except Exception as e:
            logger.error(f"时间差计算错误: {e}")


class Regularizer:
    """正则化器基类"""
    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("子类必须实现forward方法")


class L2Regularizer(Regularizer):
    """L2正则化实现"""
    def __init__(self, weight: float = 0.01):
        self.weight = weight
    
    def forward(self, factors) -> torch.Tensor:
        reg = 0.0
        for factor in factors:
            if isinstance(factor, torch.Tensor):
                reg += self.weight * torch.sum(factor ** 2)
        return reg


class L1Regularizer(Regularizer):
    """L1正则化实现"""
    def __init__(self, weight: float = 0.01):
        self.weight = weight
    
    def forward(self, factors) -> torch.Tensor:
        reg = 0.0
        for factor in factors:
            if isinstance(factor, torch.Tensor):
                reg += self.weight * torch.sum(torch.abs(factor))
        return reg


class LCGE(nn.Module):
    """
    时序知识图谱嵌入模型LCGE (Logical and Context-aware Graph Embedding)
    结合逻辑规则和时间上下文的知识图谱表示学习模型
    """
    def __init__(
            self, 
            entity_size: int, 
            relation_size: int, 
            time_size: int,
            static_size: int,
            rank: int = 200, 
            w_static: float = 0.1,
            no_time_emb: bool = False, 
            init_size: float = 1e-3,
            use_cuda: bool = True
    ):
        super(LCGE, self).__init__()
        self.entity_size = entity_size
        self.relation_size = relation_size
        self.time_size = time_size
        self.static_size = static_size
        self.rank = rank
        self.rank_static = rank // 20
        self.w_static = w_static
        self.no_time_emb = no_time_emb
        self.use_cuda = use_cuda
        
        # 定义嵌入层
        self.entity_emb = nn.Embedding(entity_size, 2 * rank, sparse=True)
        self.relation_emb = nn.Embedding(relation_size, 2 * rank, sparse=True)
        self.time_emb = nn.Embedding(time_size, 2 * rank, sparse=True)
        self.relation_no_time_emb = nn.Embedding(relation_size, 2 * rank, sparse=True)
        self.time_trans_emb = nn.Embedding(1, 2 * rank, sparse=True)  # 时间转换嵌入
        
        # 静态嵌入层
        self.entity_static_emb = nn.Embedding(entity_size, 2 * self.rank_static, sparse=True)
        self.relation_static_emb = nn.Embedding(relation_size, 2 * self.rank_static, sparse=True)
        
        # 初始化权重
        self._init_weights(init_size)
        
        # 规则相关参数 (使用字典存储以提高灵活性)
        self.rules = {
            'rule1_p2': defaultdict(dict),
            'rule2_p1': defaultdict(dict),
            'rule2_p2': defaultdict(dict),
            'rule2_p3': defaultdict(dict),
            'rule2_p4': defaultdict(dict)
        }
        
        # 设备配置
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        if use_cuda:
            self.to(self.device)
    
    def _init_weights(self, init_size: float) -> None:
        """初始化模型权重"""
        self.entity_emb.weight.data.uniform_(-init_size, init_size)
        self.relation_emb.weight.data.uniform_(-init_size, init_size)
        self.time_emb.weight.data.uniform_(-init_size, init_size)
        self.relation_no_time_emb.weight.data.uniform_(-init_size, init_size)
        self.time_trans_emb.weight.data.uniform_(-init_size, init_size)
        self.entity_static_emb.weight.data.uniform_(-init_size, init_size)
        self.relation_static_emb.weight.data.uniform_(-init_size, init_size)
    
    def _split_embedding(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """将嵌入分为实部和虚部"""
        half = embedding.size(1) // 2
        return embedding[:, :half], embedding[:, half:]
    
    def score(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算三元组的得分
        
        Args:
            x: 输入张量，形状为(batch_size, 4)，包含(lhs, rel, rhs, time)
        
        Returns:
            包含动态得分和静态得分的元组
        """
        lhs = self.entity_emb(x[:, 0])
        rel = self.relation_emb(x[:, 1])
        rel_no_time = self.relation_no_time_emb(x[:, 1])
        rhs = self.entity_emb(x[:, 2])
        time = self.time_emb(x[:, 3])
        
        # 分割嵌入
        lhs_re, lhs_im = self._split_embedding(lhs)
        rel_re, rel_im = self._split_embedding(rel)
        rhs_re, rhs_im = self._split_embedding(rhs)
        time_re, time_im = self._split_embedding(time)
        rnt_re, rnt_im = self._split_embedding(rel_no_time)
        
        # 计算时间相关关系表示
        rt_re1 = rel_re * time_re
        rt_im1 = rel_im * time_re
        rt_re2 = rel_re * time_im
        rt_im2 = rel_im * time_im
        full_rel_re = (rt_re1 - rt_im2) + rnt_re
        full_rel_im = (rt_im1 + rt_re2) + rnt_im
        
        # 静态嵌入处理
        h_static = self.entity_static_emb(x[:, 0])
        r_static = self.relation_static_emb(x[:, 1])
        t_static = self.entity_static_emb(x[:, 2])
        
        h_static_re, h_static_im = self._split_embedding(h_static)
        r_static_re, r_static_im = self._split_embedding(r_static)
        t_static_re, t_static_im = self._split_embedding(t_static)
        
        # 计算动态得分
        dynamic_score = (
            (lhs_re * full_rel_re - lhs_im * full_rel_im) * rhs_re + 
            (lhs_im * full_rel_re + lhs_re * full_rel_im) * rhs_im
        ).sum(dim=1, keepdim=True)
        
        # 计算静态得分
        static_score = (
            (h_static_re * r_static_re - h_static_im * r_static_im) * t_static_re + 
            (h_static_im * r_static_re + h_static_re * r_static_im) * t_static_im
        ).sum(dim=1, keepdim=True)
        
        return dynamic_score, static_score
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple, torch.Tensor, torch.Tensor]:
        """
        前向传播过程，包括得分计算、正则化和规则约束
        
        Args:
            x: 输入张量，形状为(batch_size, 4)，包含(lhs, rel, rhs, time)
        
        Returns:
            包含动态预测、静态预测、正则化因子、时间嵌入和规则损失的元组
        """
        x = x.to(self.device)
        lhs = self.entity_emb(x[:, 0])
        rel = self.relation_emb(x[:, 1])
        rel_no_time = self.relation_no_time_emb(x[:, 1])
        rhs = self.entity_emb(x[:, 2])
        time = self.time_emb(x[:, 3])
        transt = self.time_trans_emb(torch.LongTensor([0]).to(self.device))
        
        # 分割嵌入
        lhs_re, lhs_im = self._split_embedding(lhs)
        rel_re, rel_im = self._split_embedding(rel)
        rhs_re, rhs_im = self._split_embedding(rhs)
        time_re, time_im = self._split_embedding(time)
        transt_re, transt_im = self._split_embedding(transt)
        rnt_re, rnt_im = self._split_embedding(rel_no_time)
        
        # 计算完整关系表示
        rt_re1 = rel_re * time_re
        rt_im1 = rel_im * time_re
        rt_re2 = rel_re * time_im
        rt_im2 = rel_im * time_im
        rrt_re = rt_re1 - rt_im2
        rrt_im = rt_im1 + rt_re2
        full_rel_re = rrt_re + rnt_re
        full_rel_im = rrt_im + rnt_im
        
        # 静态嵌入处理
        h_static = self.entity_static_emb(x[:, 0])
        r_static = self.relation_static_emb(x[:, 1])
        t_static = self.entity_static_emb(x[:, 2])
        
        h_static_re, h_static_im = self._split_embedding(h_static)
        r_static_re, r_static_im = self._split_embedding(r_static)
        t_static_re, t_static_im = self._split_embedding(t_static)
        
        # 准备右侧实体嵌入用于计算
        right = self.entity_emb.weight
        right_re, right_im = self._split_embedding(right)
        
        # 准备静态右侧实体嵌入
        right_static = self.entity_static_emb.weight
        right_static_re, right_static_im = self._split_embedding(right_static)
        
        # 计算正则化因子
        regularizer = (
            math.pow(2, 1/3) * torch.sqrt(lhs_re**2 + lhs_im**2),
            torch.sqrt(rrt_re**2 + rrt_im**2),
            torch.sqrt(rnt_re**2 + rnt_im**2),
            math.pow(2, 1/3) * torch.sqrt(rhs_re**2 + rhs_im**2),
            torch.sqrt(h_static_re**2 + h_static_im**2),
            torch.sqrt(r_static_re**2 + r_static_im**2),
            torch.sqrt(t_static_re**2 + t_static_im**2)
        )
        
        # 计算规则损失
        rule_loss = self._calculate_rule_loss(x[:, 1], rel_no_time, transt_re, transt_im)
        
        # 计算动态预测
        dynamic_pred = (
            (lhs_re * full_rel_re - lhs_im * full_rel_im) @ right_re.t() +
            (lhs_im * full_rel_re + lhs_re * full_rel_im) @ right_im.t()
        )
        
        # 计算静态预测
        static_pred = (
            (h_static_re * r_static_re - h_static_im * r_static_im) @ right_static_re.t() +
            (h_static_im * r_static_re + h_static_re * r_static_im) @ right_static_im.t()
        )
        
        # 提取时间嵌入（不包含最后一个用于正则化）
        time_emb_for_reg = self.time_emb.weight[:-1] if self.no_time_emb else self.time_emb.weight
        
        return dynamic_pred, static_pred, regularizer, time_emb_for_reg, rule_loss
    
    def _calculate_rule_loss(self, rels: torch.Tensor, rel_no_time: torch.Tensor, 
                            transt_re: torch.Tensor, transt_im: torch.Tensor) -> torch.Tensor:
        """
        计算规则约束损失
        
        Args:
            rels: 关系ID张量
            rel_no_time: 无时间关系嵌入
            transt_re: 时间转换实部
            transt_im: 时间转换虚部
            
        Returns:
            规则损失张量
        """
        rule_loss = torch.tensor(0.0, device=self.device)
        rule_count = 0
        
        # 转换关系ID为列表以便处理
        rel_list = rels.cpu().numpy()
        
        for rel_idx in rel_list:
            rel_str = str(rel_idx)
            
            # 处理rule1_p2规则
            if rel_str in self.rules['rule1_p2']:
                rel1_emb = rel_no_time[rel_idx]
                rel1_re, rel1_im = self._split_embedding(rel1_emb.unsqueeze(0))
                rel1_re, rel1_im = rel1_re.squeeze(), rel1_im.squeeze()
                
                for rel2_str, weight in self.rules['rule1_p2'][rel_str].items():
                    rel2_idx = int(rel2_str)
                    rel2_emb = self.relation_no_time_emb(torch.LongTensor([rel2_idx]).to(self.device))[0]
                    rel2_re, rel2_im = self._split_embedding(rel2_emb)
                    
                    # 无时间规则损失
                    rule_loss += weight * torch.sum(torch.abs(rel1_emb - rel2_emb) ** 3)
                    rule_count += 1
                    
                    # 时间相关规则损失
                    tt_re1 = rel2_re * transt_re[0]
                    tt_im1 = rel2_im * transt_re[0]
                    tt_re2 = rel2_re * transt_im[0]
                    tt_im2 = rel2_im * transt_im[0]
                    rtt_re = tt_re1 - tt_im2
                    rtt_im = tt_im1 + tt_re2
                    
                    rule_loss += weight * (
                        torch.sum(torch.abs(rel1_re - rtt_re) ** 3) + 
                        torch.sum(torch.abs(rel1_im - rtt_im) ** 3)
                    )
                    rule_count += 1
            
            # 处理rule2系列规则
            for rule_type in ['rule2_p1', 'rule2_p2', 'rule2_p3', 'rule2_p4']:
                if rel_idx in self.rules[rule_type]:
                    rel1_re, rel1_im = self._split_embedding(rel_no_time[rel_idx].unsqueeze(0))
                    rel1_re, rel1_im = rel1_re.squeeze(), rel1_im.squeeze()
                    
                    for body, weight in self.rules[rule_type][rel_idx].items():
                        rel2_idx, rel3_idx = body
                        
                        # 获取关系嵌入
                        rel2_emb = self.relation_no_time_emb(torch.LongTensor([rel2_idx]).to(self.device))[0]
                        rel3_emb = self.relation_no_time_emb(torch.LongTensor([rel3_idx]).to(self.device))[0]
                        rel2_re, rel2_im = self._split_embedding(rel2_emb)
                        rel3_re, rel3_im = self._split_embedding(rel3_emb)
                        
                        # 根据规则类型计算不同的组合
                        if rule_type == 'rule2_p1':
                            # 复杂时间转换
                            tt2_re1 = rel2_re * transt_re[0]
                            tt2_im1 = rel2_im * transt_re[0]
                            tt2_re2 = rel2_re * transt_im[0]
                            tt2_im2 = rel2_im * transt_im[0]
                            rtt2_re = tt2_re1 - tt2_im2
                            rtt2_im = tt2_im1 + tt2_re2
                            
                            ttt2_re1 = rtt2_re * transt_re[0]
                            ttt2_im1 = rtt2_im * transt_re[0]
                            ttt2_re2 = rtt2_re * transt_im[0]
                            ttt2_im2 = rtt2_im * transt_im[0]
                            rttt2_re = ttt2_re1 - ttt2_im2
                            rttt2_im = ttt2_im1 + ttt2_re2
                            
                            tt3_re1 = rel3_re * transt_re[0]
                            tt3_im1 = rel3_im * transt_re[0]
                            tt3_re2 = rel3_re * transt_im[0]
                            tt3_im2 = rel3_im * transt_im[0]
                            rtt3_re = tt3_re1 - tt3_im2
                            rtt3_im = tt3_im1 + tt3_re2
                            
                            tt_re1 = rtt3_re * rttt2_re
                            tt_im1 = rtt3_im * rttt2_re
                            tt_re2 = rtt3_re * rttt2_im
                            tt_im2 = rtt3_im * rttt2_im
                            rtt_re = tt_re1 - tt_im2
                            rtt_im = tt_im1 + tt_re2
                        
                        elif rule_type == 'rule2_p2':
                            # 简单组合
                            tt2_re1 = rel2_re * transt_re[0]
                            tt2_im1 = rel2_im * transt_re[0]
                            tt2_re2 = rel2_re * transt_im[0]
                            tt2_im2 = rel2_im * transt_im[0]
                            rtt2_re = tt2_re1 - tt2_im2
                            rtt2_im = tt2_im1 + tt2_re2
                            
                            tt3_re1 = rel3_re * transt_re[0]
                            tt3_im1 = rel3_im * transt_re[0]
                            tt3_re2 = rel3_re * transt_im[0]
                            tt3_im2 = rel3_im * transt_im[0]
                            rtt3_re = tt3_re1 - tt3_im2
                            rtt3_im = tt3_im1 + tt3_re2
                            
                            tt_re1 = rtt3_re * rtt2_re
                            tt_im1 = rtt3_im * rtt2_re
                            tt_re2 = rtt3_re * rtt2_im
                            tt_im2 = rtt3_im * rtt2_im
                            rtt_re = tt_re1 - tt_im2
                            rtt_im = tt_im1 + tt_re2
                        
                        elif rule_type == 'rule2_p3':
                            # 简化组合
                            tt2_re1 = rel2_re * transt_re[0]
                            tt2_im1 = rel2_im * transt_re[0]
                            tt2_re2 = rel2_re * transt_im[0]
                            tt2_im2 = rel2_im * transt_im[0]
                            rtt2_re = tt2_re1 - tt2_im2
                            rtt2_im = tt2_im1 + tt2_re2
                            
                            tt_re1 = rel3_re * rtt2_re
                            tt_im1 = rel3_im * rtt2_re
                            tt_re2 = rel3_re * rtt2_im
                            tt_im2 = rel3_im * rtt2_im
                            rtt_re = tt_re1 - tt_im2
                            rtt_im = tt_im1 + tt_re2
                        
                        else:  # rule2_p4
                            # 直接组合
                            tt_re1 = rel3_re * rel2_re
                            tt_im1 = rel3_im * rel2_re
                            tt_re2 = rel3_re * rel2_im
                            tt_im2 = rel3_im * rel2_im
                            rtt_re = tt_re1 - tt_im2
                            rtt_im = tt_im1 + tt_re2
                        
                        # 累加规则损失
                        rule_loss += weight * (
                            torch.sum(torch.abs(rel1_re - rtt_re) ** 3) + 
                            torch.sum(torch.abs(rel1_im - rtt_im) ** 3)
                        )
                        rule_count += 1
        
        # 平均规则损失
        if rule_count > 0:
            rule_loss = rule_loss / rule_count
        return rule_loss
    
    def forward_over_time(self, x: torch.Tensor) -> torch.Tensor:
        """
        跨时间的前向传播，用于时间相关预测
        
        Args:
            x: 输入张量，形状为(batch_size, 4)，包含(lhs, rel, rhs, time)
            
        Returns:
            时间相关得分
        """
        x = x.to(self.device)
        lhs = self.entity_emb(x[:, 0])
        rel = self.relation_emb(x[:, 1])
        rhs = self.entity_emb(x[:, 2])
        time_emb = self.time_emb.weight
        
        # 分割嵌入
        lhs_re, lhs_im = self._split_embedding(lhs)
        rel_re, rel_im = self._split_embedding(rel)
        rhs_re, rhs_im = self._split_embedding(rhs)
        time_re, time_im = self._split_embedding(time_emb)
        
        rel_no_time = self.relation_no_time_emb(x[:, 1])
        rnt_re, rnt_im = self._split_embedding(rel_no_time)
        
        # 计算时间相关得分
        score_time = (
            (lhs_re * rel_re * rhs_re - lhs_im * rel_im * rhs_re -
             lhs_im * rel_re * rhs_im + lhs_re * rel_im * rhs_im) @ time_re.t() +
            (lhs_im * rel_re * rhs_re - lhs_re * rel_im * rhs_re +
             lhs_re * rel_re * rhs_im - lhs_im * rel_im * rhs_im) @ time_im.t()
        )
        
        # 计算基础得分
        base_score = torch.sum(
            (lhs_re * rnt_re * rhs_re - lhs_im * rnt_im * rhs_re -
             lhs_im * rnt_re * rhs_im + lhs_re * rnt_im * rhs_im) +
            (lhs_im * rnt_im * rhs_re - lhs_re * rnt_re * rhs_re +
             lhs_re * rnt_im * rhs_im - lhs_im * rnt_re * rhs_im),
            dim=1, keepdim=True
        )
        
        return score_time + base_score
    
    def get_rhs(self, chunk_begin: int, chunk_size: int) -> torch.Tensor:
        """获取右侧实体嵌入"""
        return self.entity_emb.weight[chunk_begin:chunk_begin + chunk_size].t().to(self.device)
    
    def get_rhs_static(self, chunk_begin: int, chunk_size: int) -> torch.Tensor:
        """获取右侧实体静态嵌入"""
        return self.entity_static_emb.weight[chunk_begin:chunk_begin + chunk_size].t().to(self.device)
    
    def get_queries(self, queries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取查询的嵌入表示"""
        queries = queries.to(self.device)
        lhs = self.entity_emb(queries[:, 0])
        rel = self.relation_emb(queries[:, 1])
        rel_no_time = self.relation_no_time_emb(queries[:, 1])
        time = self.time_emb(queries[:, 3])
        
        # 分割嵌入
        lhs_re, lhs_im = self._split_embedding(lhs)
        rel_re, rel_im = self._split_embedding(rel)
        time_re, time_im = self._split_embedding(time)
        rnt_re, rnt_im = self._split_embedding(rel_no_time)
        
        # 计算完整关系表示
        rt_re1 = rel_re * time_re
        rt_im1 = rel_im * time_re
        rt_re2 = rel_re * time_im
        rt_im2 = rel_im * time_im
        full_rel_re = (rt_re1 - rt_im2) + rnt_re
        full_rel_im = (rt_im1 + rt_re2) + rnt_im
        
        # 静态嵌入处理
        h_static = self.entity_static_emb(queries[:, 0])
        r_static = self.relation_static_emb(queries[:, 1])
        
        h_static_re, h_static_im = self._split_embedding(h_static)
        r_static_re, r_static_im = self._split_embedding(r_static)
        
        # 合并表示
        dynamic_query = torch.cat([
            lhs_re * full_rel_re - lhs_im * full_rel_im,
            lhs_im * full_rel_re + lhs_re * full_rel_im
        ], 1)
        
        static_query = torch.cat([
            h_static_re * r_static_re - h_static_im * r_static_im,
            h_static_im * r_static_re + h_static_re * r_static_im
        ], 1)
        
        return dynamic_query, static_query
    
    def load_rules(self, rule_file: str) -> None:
        """从文件加载规则"""
        try:
            with open(rule_file, 'rb') as f:
                rules = pickle.load(f)
            for rule_type, rule_data in rules.items():
                if rule_type in self.rules:
                    self.rules[rule_type].update(rule_data)
            logger.info(f"成功从{rule_file}加载规则")
        except FileNotFoundError:
            logger.warning(f"规则文件{rule_file}不存在，使用空规则")
        except Exception as e:
            logger.error(f"加载规则时出错: {e}")
    
    def save_model(self, model_name: str) -> None:
        """保存模型"""
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), MODEL_PATH / f"{model_name}.pt")
        logger.info(f"模型已保存至{MODEL_PATH / f'{model_name}.pt'}")
    
    @classmethod
    def load_model(cls, model_path: str, entity_size: int, relation_size: int, 
                  time_size: int, static_size: int, rank: int = 200, 
                  w_static: float = 0.1, no_time_emb: bool = False,
                  use_cuda: bool = True) -> 'LCGE':
        """加载模型"""
        model = cls(entity_size, relation_size, time_size, static_size, rank, w_static, no_time_emb, use_cuda=use_cuda)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")))
        model.to(torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu"))
        logger.info(f"模型已从{model_path}加载")
        return model


class TKBTrainer:
    """时序知识图谱训练器"""
    def __init__(
            self, 
            model: LCGE,
            emb_regularizer: Regularizer,
            temporal_regularizer: Regularizer,
            rule_regularizer: Regularizer,
            optimizer: Optimizer,
            train_dataset: TemporalKnowledgeGraphDataset,
            valid_dataset: TemporalKnowledgeGraphDataset,
            batch_size: int = 256,
            epochs: int = 100,
            lr_scheduler: Optional = None,
            early_stopping_patience: int = 10,
            verbose: bool = True
    ):
        self.model = model
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.rule_regularizer = rule_regularizer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose
        self.best_valid_loss = float('inf')
        self.patience = 0
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        self.valid_loader = DataLoader(
            valid_dataset, 
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True
        )
    
    def train_epoch(self) -> float:
        """执行一个训练轮次"""
        self.model.train()
        total_loss = 0.0
        loss_fn = nn.CrossEntropyLoss(reduction='mean')
        loss_static_fn = nn.CrossEntropyLoss(reduction='mean')
        
        with tqdm(self.train_loader, disable=not self.verbose) as bar:
            bar.set_description('训练中')
            for batch in bar:
                # 移动到设备
                batch = batch.to(self.model.device)
                targets = batch[:, 2]  # 右侧实体ID
                
                # 前向传播
                dynamic_pred, static_pred, factors, time_emb, rule_loss = self.model(batch)
                
                # 计算损失
                fit_loss = loss_fn(dynamic_pred, targets)
                static_loss = loss_static_fn(static_pred, targets)
                emb_reg = self.emb_regularizer.forward(factors)
                temp_reg = self.temporal_regularizer.forward(time_emb)
                rule_reg = self.rule_regularizer.forward(rule_loss)
                
                # 总损失
                total_loss_batch = fit_loss + 0.1 * static_loss + emb_reg + temp_reg + rule_reg
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                total_loss_batch.backward()
                self.optimizer.step()
                
                total_loss += total_loss_batch.item() * batch.size(0)
                
                # 更新进度条
                bar.set_postfix(
                    loss=f'{fit_loss.item():.4f}',
                    loss_cs=f'{static_loss.item():.4f}',
                    reg=f'{emb_reg.item():.4f}',
                    cont=f'{temp_reg.item():.4f}',
                    rule=f'{rule_reg.item():.4f}'
                )
        
        return total_loss / len(self.train_dataset)
    
    def validate(self) -> float:
        """执行验证"""
        self.model.eval()
        total_loss = 0.0
        loss_fn = nn.CrossEntropyLoss(reduction='mean')
        loss_static_fn = nn.CrossEntropyLoss(reduction='mean')
        
        with torch.no_grad(), tqdm(self.valid_loader, disable=not self.verbose) as bar:
            bar.set_description('验证中')
            for batch in bar:
                # 移动到设备
                batch = batch.to(self.model.device)
                targets = batch[:, 2]  # 右侧实体ID
                
                # 前向传播
                dynamic_pred, static_pred, factors, time_emb, rule_loss = self.model(batch)
                
                # 计算损失
                fit_loss = loss_fn(dynamic_pred, targets)
                static_loss = loss_static_fn(static_pred, targets)
                emb_reg = self.emb_regularizer.forward(factors)
                temp_reg = self.temporal_regularizer.forward(time_emb)
                rule_reg = self.rule_regularizer.forward(rule_loss)
                
                # 总损失
                total_loss_batch = fit_loss + 0.1 * static_loss + emb_reg + temp_reg + rule_reg
                
                total_loss += total_loss_batch.item() * batch.size(0)
                
                # 更新进度条
                bar.set_postfix(
                    loss=f'{fit_loss.item():.4f}',
                    loss_cs=f'{static_loss.item():.4f}',
                    reg=f'{emb_reg.item():.4f}',
                    cont=f'{temp_reg.item():.4f}',
                    rule=f'{rule_reg.item():.4f}'
                )
        
        return total_loss / len(self.valid_dataset)
    
    def train(self, model_name: str) -> None:
        """执行完整的训练过程"""
        logger.info(f"开始训练模型，批次大小: {self.batch_size}, 轮次: {self.epochs}")
        
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()
            valid_loss = self.validate()
            
            logger.info(f"Epoch {epoch}/{self.epochs} - 训练损失: {train_loss:.4f}, 验证损失: {valid_loss:.4f}")
            
            # 更新学习率
            if self.lr_scheduler:
                self.lr_scheduler.step(valid_loss)
            
            # 早停检查
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.model.save_model(model_name)
                self.patience = 0
                logger.info(f"保存最佳模型，验证损失: {valid_loss:.4f}")
            else:
                self.patience += 1
                if self.patience >= self.early_stopping_patience:
                    logger.info(f"早停触发，耐心值: {self.patience}")
                    break
        
        logger.info(f"训练完成，最佳验证损失: {self.best_valid_loss:.4f}")


# 主函数示例
def main():
    # 配置ICEWS数据集预处理
    icews_config = DatasetConfig(
        name="ICEWS05-15",
        path="./data/ICEWS05-15",
        train_file="train",
        valid_file="valid",
        test_file="test"
    )
    
    # 执行ICEWS数据集预处理
    icews_preprocessor = ICEWSpreprocessor(icews_config)
    icews_preprocessor.preprocess()
    
    # 配置wiki12K数据集预处理
    wiki_config = DatasetConfig(
        name="wiki12K",
        path="./data/wiki12K",
        train_file="train",
        valid_file="valid",
        test_file="test",
        has_time=True
    )
    
    # 执行wiki12K数据集预处理
    wiki_preprocessor = Wiki12KPreprocessor(wiki_config)
    wiki_preprocessor.preprocess()
    
    # 加载预处理后的数据集
    with open(DATA_PATH / "ICEWS05-15" / "train.pickle", 'rb') as f:
        train_data = pickle.load(f)
    with open(DATA_PATH / "ICEWS05-15" / "valid.pickle", 'rb') as f:
        valid_data = pickle.load(f)
    
    # 创建数据集对象
    train_dataset = TemporalKnowledgeGraphDataset(train_data)
    valid_dataset = TemporalKnowledgeGraphDataset(valid_data)
    
    # 获取实体和关系数量
    with open(DATA_PATH / "ICEWS05-15" / "ent_id", 'r') as f:
        entity_size = len(f.readlines())
    with open(DATA_PATH / "ICEWS05-15" / "rel_id", 'r') as f:
        relation_size = len(f.readlines())
    with open(DATA_PATH / "ICEWS05-15" / "ts_id", 'r') as f:
        time_size = len(f.readlines())
    
    # 初始化模型
    model = LCGE(
        entity_size=entity_size,
        relation_size=relation_size,
        time_size=time_size,
        static_size=100,
        rank=200,
        w_static=0.1,
        no_time_emb=False,
        use_cuda=True
    )
    
    # 定义优化器和正则化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    emb_regularizer = L2Regularizer(weight=0.001)
    temporal_regularizer = L2Regularizer(weight=0.001)
    rule_regularizer = L2Regularizer(weight=0.001)
    
    # 定义学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 创建训练器并开始训练
    trainer = TKBTrainer(
        model=model,
        emb_regularizer=emb_regularizer,
        temporal_regularizer=temporal_regularizer,
        rule_regularizer=rule_regularizer,
        optimizer=optimizer,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        batch_size=256,
        epochs=50,
        lr_scheduler=lr_scheduler,
        early_stopping_patience=10,
        verbose=True
    )
    
    trainer.train("lcge_icews_model")


if __name__ == "__main__":
    main()
