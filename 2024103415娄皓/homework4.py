import os
from time import time, strftime, localtime
from tqdm import tqdm
import json
import torch
import shutil
import random
import argparse
import numpy as np
import itertools
from transformers import BertTokenizer, BertForMaskedLM, BertConfig, get_cosine_schedule_with_warmup
from contiguous_params import ContiguousParams
from sklearn.decomposition import PCA

from Code import KGCDataModule
from Code import NBert, NFormer, Knowformer, Inter_Classifier
from utils import save_model, load_model, score2str
from torch.optim.adamw import AdamW
import matplotlib.pyplot as plt

# 设置CUDA设备相关的环境变量
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def set_result_path(args, name):
    """
    设置结果保存路径
    Args:
        args: 参数字典
        name: 结果路径的名称标识
    Returns:
        result_dir: 结果保存路径
    """
    root_path = os.path.dirname(__file__)
    result_dir = os.path.join(root_path, 'result', args['dataset'], name, args['complex'],
                              str(int(args['anomaly_ratio'] * 100)))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir


def get_args():
    """
    获取命令行参数
    Returns:
        args: 参数字典
    """
    parser = argparse.ArgumentParser()
    # 训练相关参数
    parser.add_argument('--task', type=str, default='train', help='任务类型')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--epoch', type=int, default=20, help='训练的轮数')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--device', type=str, default='cuda:3', help='使用的设备')
    parser.add_argument('--dataset', type=str, default='fb15k-237', help='使用的数据集')
    parser.add_argument('--max_seq_length', type=int, default=64, help='输入序列的最大长度')
    # 邻居相关参数
    parser.add_argument('--extra_encoder', action='store_true', default=False)
    parser.add_argument('--add_neighbors', action='store_true', default=True)
    parser.add_argument('--neighbor_num', type=int, default=3)
    parser.add_argument('--neighbor_token', type=str, default='[Neighbor]')
    parser.add_argument('--no_relation_token', type=str, default='[R_None]')
    # 文本编码器相关参数
    parser.add_argument('--lm_lr', type=float, default=1e-4, help='语言模型的学习率')
    parser.add_argument('--lm_label_smoothing', type=float, default=0.8, help='语言模型的标签平滑参数')
    # 结构编码器相关参数
    parser.add_argument('--kge_lr', type=float, default=7e-4)
    parser.add_argument('--kge_label_smoothing', type=float, default=0.8)
    parser.add_argument('--num_hidden_layers', type=int, default=6)
    parser.add_argument('--num_attention_heads', type=int, default=4)
    parser.add_argument('--input_dropout_prob', type=float, default=0.1, help='输入的丢弃概率')
    parser.add_argument('--context_dropout_prob', type=float, default=0.1, help='上下文的丢弃概率')
    parser.add_argument('--attention_dropout_prob', type=float, default=0.3)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--intermediate_size', type=int, default=2048)
    parser.add_argument('--residual_dropout_prob', type=float, default=0.0)
    parser.add_argument('--initializer_range', type=float, default=0.02)
    # 其他参数
    parser.add_argument('--num_workers', type=int, default=32, help='DataLoader使用的工人数量')
    parser.add_argument('--pin_memory', type=bool, default=True, help='是否固定内存')
    parser.add_argument('--scheme', type=str, default='mlp')
    parser.add_argument('--contrastive_lr', type=float, default=1e-4, help='对比模型的学习率')
    parser.add_argument('--concatenate_lr', type=float, default=1e-4, help='拼接模型的学习率')
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--use_bert', action='store_true', default=True)
    parser.add_argument('--use_transformer', action='store_true', default=True)
    parser.add_argument('--use_joint_training', action='store_true', default=True)
    parser.add_argument('--use_concatenate', action='store_true', default=False)
    parser.add_argument('--use_contrastive', action='store_true', default=True)
    parser.add_argument('--use_kgc', action='store_true', default=True)

    args = parser.parse_args()
    args = vars(args)

    # 设置额外的配置项
    root_path = os.path.dirname(__file__)
    args['tokenizer_path'] = os.path.join(root_path, 'checkpoints', 'bert-base-cased')
    args['data_path'] = os.path.join(root_path, 'dataset', args['dataset'])
    args['complex'] = 'mixture_anomaly'
    args['anomaly_ratio'] = 0.05
    args['final_result_path'] = set_result_path(args, 'final_result')

    # 设置随机种子以确保结果可复现
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    return args


def get_model_config(config):
    """
    获取模型配置
    Args:
        config: 配置字典
    Returns:
        model_config: 模型配置字典
    """
    model_config = dict()
    model_config["hidden_size"] = config['hidden_size']
    model_config["num_hidden_layers"] = config['num_hidden_layers']
    model_config["num_attention_heads"] = config['num_attention_heads']
    model_config["input_dropout_prob"] = config['input_dropout_prob']
    model_config["attention_dropout_prob"] = config['attention_dropout_prob']
    model_config["hidden_dropout_prob"] = config['hidden_dropout_prob']
    model_config["residual_dropout_prob"] = config['residual_dropout_prob']
    model_config["context_dropout_prob"] = config['context_dropout_prob']
    model_config["initializer_range"] = config['initializer_range']
    model_config["intermediate_size"] = config['intermediate_size']

    model_config["vocab_size"] = config['vocab_size']
    model_config["num_relations"] = config['num_relations']

    model_config['device'] = config['device']
    return model_config


class CCATrainer:
    def __init__(self, config: dict):
        """
        初始化CCATrainer类
        Args:
            config: 配置字典
        """
        self.epoch = config['epoch']

        tokenizer, self.train_dl, self.label = self._load_dataset(config)
        self.model_config, self.text_model, self.struc_model, self.inter_classifier = self._load_model(config,
                                                                                                       tokenizer)

        self.final_result_path = config['final_result_path']

        self.return_all_layer = False

        opt3 = self.configure_optimizers(total_steps=len(self.train_dl) * self.epoch)
        self.joint_opt, self.joint_sche = opt3['optimizer'], opt3['scheduler']

        self.scaler = torch.cuda.amp.GradScaler() if config['use_amp'] else None
        self.low_degree = config['low_degree']
        self.struc_soft_label = None
        self.text_soft_label = None
        self.inter_soft_label = None

    def _load_dataset(self, config: dict):
        """
        加载数据集
        Args:
            config: 配置字典
        Returns:
            tokenizer: 分词器
            train_dl: 训练数据加载器
            label: 标签信息
        """
        # 加载分词器
        tokenizer_path = config['tokenizer_path']
        print(f'从 {tokenizer_path} 加载分词器')
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_basic_tokenize=False, use_fast=True)
        # 加载数据模块
        data_module = KGCDataModule(config, tokenizer, encode_text=True, encode_struc=True)
        config['low_degree'] = data_module.low_degree
        tokenizer = data_module.get_tokenizer()
        train_dl = data_module.get_train_dataloader()
        label = data_module.label
        return tokenizer, train_dl, label

    def _load_model(self, config: dict, tokenizer: BertTokenizer):
        """
        加载模型
        Args:
            config: 配置字典
            tokenizer: 分词器
        Returns:
            model_config: 模型配置
            text_model: 文本模型
            struc_model: 结构模型
            inter_classifier: 交互分类器
        """
        # 加载文本模型
        text_encoder_path = config['model_path']
        print(f'从 {text_encoder_path} 加载Bert模型')
        bert_encoder = BertForMaskedLM.from_pretrained(text_encoder_path)
        text_model = NBert(config, tokenizer, bert_encoder).to(config['device'])
        # 加载结构模型
        model_config = get_model_config(config)
        bert_encoder = Knowformer(model_config)
        struc_model = NFormer(config, bert_encoder).to(config['device'])
        # 加载交互分类器
        inter_classifier = Inter_Classifier(config).to(config['device'])
        return model_config, text_model, struc_model, inter_classifier

    def _train_one_epoch(self, epoch):
        """
        训练一个epoch
        Args:
            epoch: 当前epoch
        """
        self.text_model.train()
        self.struc_model.train()
        struc_output = list()
        text_output = list()
        if self.return_all_layer:
            struc_loss_info = {k: [] for k in range(6)}
        else:
            struc_loss_info = []
        text_loss_info = []
        inter_loss_info = []
        joint_loss_info = []
        all_inter_logits = []

        batch_struc_logits_score = []
        batch_text_logits_score = []
        # 设置软标签
        bert_soft_label = self.inter_soft_label
        transformer_soft_label = self.inter_soft_label
        inter_soft_label = self.inter_soft_label
        # KGC邻居标签使用KGScore排名
        neighbor_soft_label = self.struc_soft_label
        bar = tqdm(self.train_dl)
        threshold = 4
        for batch_idx, batch_data in enumerate(bar):
            if epoch > threshold:
                batch_data['soft_labels'] = torch.tensor([bert_soft_label[code] for code in batch_data['code']])
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    text_batch_loss, text_info, text_logits, text_logits_score = self.text_model.training_step(
                        batch_data, batch_idx)
            else:
                text_batch_loss, text_info, text_logits, text_logits_score = self.text_model.training_step(batch_data,
                                                                                                           batch_idx)
            text_loss_info += text_info
            batch_text_logits_score += text_logits_score

            if epoch > threshold:
                batch_data['soft_labels'] = torch.tensor([transformer_soft_label[code] for code in batch_data['code']])
            if False:
                batch_data['head_neighbor_trustworthy'] = torch.tensor(list(
                    map(lambda x: list(map(lambda i: 0 if i == -1 else neighbor_soft_label[i], x)),
                        batch_data['head_struc_neighbors_code'])))
                batch_data['tail_neighbor_trustworthy'] = torch.tensor(list(
                    map(lambda x: list(map(lambda i: 0 if i == -1 else neighbor_soft_label[i], x)),
                        batch_data['tail_struc_neighbors_code'])))

            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    struc_batch_loss, struc_info, struc_logits, struc_logits_score = self.struc_model.training_step(
                        batch_data, self.return_all_layer)
            else:
                struc_batch_loss, struc_info, struc_logits, struc_logits_score = self.struc_model.training_step(
                    batch_data, self.return_all_layer)
            if self.return_all_layer:
                for i in range(len(struc_info)):
                    struc_loss_info[i] += struc_info[i]
            else:
                struc_loss_info += struc_info
                batch_struc_logits_score += struc_logits_score

            if epoch > threshold:
                batch_data['soft_labels'] = torch.tensor([self.inter_soft_label[code] for code in batch_data['code']])
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    inter_batch_loss, inter_info, inter_logits = self.inter_classifier(text_logits, struc_logits,
                                                                                       batch_data, text_logits_score,
                                                                                       struc_logits_score)
            else:
                inter_batch_loss, inter_info, inter_logits = self.inter_classifier(text_logits, struc_logits,
                                                                                   batch_data, text_logits_score,
                                                                                   struc_logits_score)
            inter_loss_info += inter_info

            joint_batch_loss = text_batch_loss + struc_batch_loss + inter_batch_loss
            self.joint_opt.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(joint_batch_loss).backward()
                self.scaler.unscale_(self.joint_opt)
                self.scaler.step(self.joint_opt)
                self.scaler.update()
            else:
                joint_batch_loss.backward()
                self.joint_opt.step()
            if self.joint_sche is not None:
                self.joint_sche.step()
            joint_loss_info += inter_info
            self.joint_opt.zero_grad()
        return text_loss_info, struc_loss_info, inter_loss_info, batch_text_logits_score, batch_struc_logits_score

    def train(self):
        """
        训练模型
        """
        for i in range(1, self.epoch + 1):
            begin_time = time()
            text_loss_info, struc_loss_info, inter_loss_info, text_logits_score, struc_logits_score = self._train_one_epoch(
                i)
            # 更新软标签并验证
            if self.low_degree:
                inter_rank = self.score_function2(text_logits_score, struc_logits_score, inter_loss_info)
                inter_rank, self.inter_soft_label = self.update_soft_label(inter_rank, i)
                self.validate(inter_rank, i, self.final_result_path)
            else:
                inter_rank = self.score_function3(text_loss_info, struc_loss_info, inter_loss_info)
                inter_rank, self.inter_soft_label = self.update_soft_label(inter_rank, i)
                self.validate(inter_rank, i, self.final_result_path)

    def score_function2(self, text_logits_score, struc_logits_score, inter_score=None, equal=False):
        """
        计算分数函数2
        Args:
            text_logits_score: 文本逻辑分数
            struc_logits_score: 结构逻辑分数
            inter_score: 交互分数
            equal: 是否平等
        Returns:
            final_score: 最终分数
        """

        def takeSecond(elem):
            return elem[1]

        construction_score = {}
        text_logits_score = dict(text_logits_score)
        struc_logits_score = dict(struc_logits_score)
        for code in text_logits_score.keys():
            if inter_score is not None:
                construction_score[code] = - (0.2 * struc_logits_score[code] + text_logits_score[code])
        construction_score = list(construction_score.items())
        construction_score.sort(key=takeSecond, reverse=True)
        inter_score.sort(key=takeSecond, reverse=True)
        term = 0.005 * len(inter_score)
        contrastive_bias = 3
        inter_score = [(inter_score[i][0], 1 / ((int(i / (term) + contrastive_bias) ** contrastive_bias))) for i in
                       range(len(inter_score))]
        construction_score = [(construction_score[i][0], 1 / ((int(i / (term)) + 1))) for i in
                              range(len(construction_score))]
        final_score = {}
        inter_score = dict(inter_score)
        construction_score = dict(construction_score)
        for i in inter_score.keys():
            final_score[i] = inter_score[i] + construction_score[i]
        final_score = list(final_score.items())
        final_score.sort(key=takeSecond, reverse=True)
        return final_score

    def score_function3(self, text_logits_score, struc_logits_score, inter_score=None, equal=False):
        """
        计算分数函数3
        Args:
            text_logits_score: 文本逻辑分数
            struc_logits_score: 结构逻辑分数
            inter_score: 交互分数
            equal: 是否平等
        Returns:
            final_score: 最终分数
        """

        def takeSecond(elem):
            return elem[1]

        text_logits_score.sort(key=takeSecond, reverse=True)
        struc_logits_score.sort(key=takeSecond, reverse=True)
        inter_score.sort(key=takeSecond, reverse=True)
        term = 0.005 * len(inter_score)
        contrastive_bias = 2
        struc_bias = 1.5
        inter_score = [(inter_score[i][0], 1 / ((int(i / (term)) ** contrastive_bias + 4))) for i in
                       range(len(inter_score))]
        text_score = [(text_logits_score[i][0], 1 / ((int(i / (term)) + 1))) for i in range(len(text_logits_score))]
        struc_score = [(struc_logits_score[i][0], 1 / ((int(i / (term)) ** struc_bias + 4))) for i in
                       range(len(struc_logits_score))]
        final_score = {}
        inter_score = dict(inter_score)
        text_score = dict(text_score)
        struc_score = dict(struc_score)
        for i in inter_score.keys():
            final_score[i] = inter_score[i] + text_score[i] + struc_score[i]
        final_score = list(final_score.items())
        final_score.sort(key=takeSecond, reverse=True)
        return final_score

    def validate(self, rank, epoch, path):
        """
        验证模型
        Args:
            rank: 排名
            epoch: 当前epoch
            path: 路径
        """
        truth = dict(self.label)
        correct_len = sum([x[0] for x in truth.values()])
        anomaly_len = len(rank) - correct_len
        print('异常长度:' + str(anomaly_len))
        topK = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15,
                0.20, 0.30, 0.4, 0.5, 0.6, 0.7, 0.8]
        numK = list(map(int, topK * len(rank)))
        result_path = os.path.join(path, str(epoch) + '.txt')
        predict_path = os.path.join(path, str(epoch) + '_predict' + '.txt')

        with open(result_path, 'w') as f:
            for top in topK:
                tp = 0
                fp = 0
                num_k = int(len(rank) * top)
                for i in range(num_k):
                    code = rank[i][0]
                    if truth[code][0] == 0:
                        tp += 1
                    else:
                        fp += 1
                recall = tp * 1.0 / anomaly_len
                precision = tp * 1.0 / num_k
                print('epoch: %d, Top%f: precision: %f, recall %f:' % (epoch, top, precision, recall))
                f.write('epoch: %d, Top%f: precision: %f, recall %f:\n' % (epoch, top, precision, recall))
        signal = 0
        with open(predict_path, 'w') as f:
            for i in range(len(rank)):
                code = rank[i][0]
                if i == numK[signal]:
                    f.write('#' + '\t' + 'top' + '\t' + str(topK[signal]) + '\n')
                    signal = (signal + 1) % len(numK)
                if truth[code][0] == 0:
                    triple = truth[code][1]
                    f.write('anomaly:\t' + triple[0] + '\t' + triple[1] + '\t' + triple[2] + '\n')
                else:
                    triple = truth[code][1]
                    f.write(triple[0] + '\t' + triple[1] + '\t' + triple[2] + '\n')
        return rank

    def update_soft_label(self, sample_loss, epoch):
        """
        更新软标签
        Args:
            sample_loss: 样本损失
            epoch: 当前epoch
        Returns:
            rank: 排名
            soft_label: 软标签
        """

        def takeSecond(elem):
            return elem[1]

        rank = []
        rank2 = []
        # 归一化
        # 删除负样本
        for loss in sample_loss:
            if loss[0] != -1:
                rank.append(loss)
        # 按损失排序
        temperature = 1
        rank.sort(key=takeSecond, reverse=True)
        t = 1
        loss_list = [loss[1] for loss in rank]

        x = list(np.random.normal(loc=0.5, scale=1, size=(len(loss_list))))
        x.sort(reverse=True)
        weight_list = 1 - t * (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-9)

        soft_label = {key: 0 for key in range(len(sample_loss))}
        term = int(0.01 * len(rank))
        for i in range(len(rank)):
            code = rank[i][0]
            weight = 1 - 1 / (int(i / term) + 1)
            soft_label[code] = weight
        soft_label[-1] = 0
        return rank, soft_label

    def configure_optimizers(self, total_steps: int):
        """
        配置优化器
        Args:
            total_steps: 总步骤数
        Returns:
            optimizer_scheduler: 优化器和调度器
        """
        parameters = self.get_parameters()
        opt = AdamW(parameters, eps=1e-6)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=opt,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        return {'optimizer': opt, 'scheduler': scheduler}

    def get_parameters(self):
        """
        获取参数
        Returns:
            final_param: 最终参数
        """
        final_param = []
        text_encoder_param = self.text_model.get_parameters()
        struc_encoder_param = self.struc_model.get_parameters()
        inter_classifier_param = self.inter_classifier.get_parameters()
        final_param = itertools.chain(text_encoder_param, struc_encoder_param, inter_classifier_param)
        return final_param

    def main(self):
        """
        主函数
        """
        self.train()


if __name__ == '__main__':
    config = get_args()
    trainer = CCATrainer(config)
    trainer.main()