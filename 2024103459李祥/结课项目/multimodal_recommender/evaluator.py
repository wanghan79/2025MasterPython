"""
评估模块
包含各种推荐系统评估指标的计算
"""

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from collections import defaultdict


class RecommenderEvaluator:
    """推荐系统评估器"""
    
    def __init__(self, top_k_list=[5, 10, 20]):
        self.top_k_list = top_k_list
        self.reset()
    
    def reset(self):
        """重置评估器"""
        self.predictions = []
        self.targets = []
        self.user_ids = []
        self.item_ids = []
    
    def add_batch(self, predictions, targets, user_ids, item_ids):
        """添加一个批次的预测结果"""
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.user_ids.extend(user_ids.cpu().numpy())
        self.item_ids.extend(item_ids.cpu().numpy())
    
    def compute_regression_metrics(self):
        """计算回归指标"""
        if len(self.predictions) == 0:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mse)
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse
        }
    
    def compute_ranking_metrics(self):
        """计算排序指标"""
        if len(self.predictions) == 0:
            return {}
        
        # 构建用户-物品评分字典
        user_item_scores = defaultdict(list)
        user_item_targets = defaultdict(list)
        
        for i, (user_id, item_id, pred, target) in enumerate(
            zip(self.user_ids, self.item_ids, self.predictions, self.targets)
        ):
            user_item_scores[user_id].append((item_id, pred))
            user_item_targets[user_id].append((item_id, target))
        
        metrics = {}
        
        for k in self.top_k_list:
            precision_scores = []
            recall_scores = []
            ndcg_scores = []
            hit_ratios = []
            
            for user_id in user_item_scores:
                # 获取该用户的预测分数和真实分数
                scores = user_item_scores[user_id]
                targets = dict(user_item_targets[user_id])
                
                # 按预测分数排序
                scores.sort(key=lambda x: x[1], reverse=True)
                
                # 获取top-k推荐
                top_k_items = [item_id for item_id, _ in scores[:k]]
                
                # 计算相关物品（评分>=4的物品）
                relevant_items = [item_id for item_id, score in targets.items() if score >= 4.0]
                
                if len(relevant_items) == 0:
                    continue
                
                # 计算指标
                hit_items = set(top_k_items) & set(relevant_items)
                
                # Precision@K
                precision = len(hit_items) / k if k > 0 else 0
                precision_scores.append(precision)
                
                # Recall@K
                recall = len(hit_items) / len(relevant_items) if len(relevant_items) > 0 else 0
                recall_scores.append(recall)
                
                # Hit Ratio@K
                hit_ratio = 1 if len(hit_items) > 0 else 0
                hit_ratios.append(hit_ratio)
                
                # NDCG@K
                ndcg = self._compute_ndcg(top_k_items, relevant_items, targets, k)
                ndcg_scores.append(ndcg)
            
            # 计算平均值
            metrics[f'Precision@{k}'] = np.mean(precision_scores) if precision_scores else 0
            metrics[f'Recall@{k}'] = np.mean(recall_scores) if recall_scores else 0
            metrics[f'NDCG@{k}'] = np.mean(ndcg_scores) if ndcg_scores else 0
            metrics[f'Hit_Ratio@{k}'] = np.mean(hit_ratios) if hit_ratios else 0
        
        return metrics
    
    def _compute_ndcg(self, recommended_items, relevant_items, targets, k):
        """计算NDCG@K"""
        dcg = 0
        for i, item_id in enumerate(recommended_items[:k]):
            if item_id in relevant_items:
                relevance = targets[item_id]
                dcg += (2 ** relevance - 1) / np.log2(i + 2)
        
        # 计算IDCG
        relevant_scores = [targets[item_id] for item_id in relevant_items]
        relevant_scores.sort(reverse=True)
        
        idcg = 0
        for i, relevance in enumerate(relevant_scores[:k]):
            idcg += (2 ** relevance - 1) / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0
    
    def compute_all_metrics(self):
        """计算所有评估指标"""
        regression_metrics = self.compute_regression_metrics()
        ranking_metrics = self.compute_ranking_metrics()
        
        all_metrics = {**regression_metrics, **ranking_metrics}
        return all_metrics
    
    def print_metrics(self, metrics=None):
        """打印评估指标"""
        if metrics is None:
            metrics = self.compute_all_metrics()
        
        print("=" * 60)
        print("评估结果")
        print("=" * 60)
        
        # 回归指标
        print("回归指标:")
        for metric in ['MSE', 'MAE', 'RMSE']:
            if metric in metrics:
                print(f"  {metric}: {metrics[metric]:.4f}")
        
        print()
        
        # 排序指标
        print("排序指标:")
        for k in self.top_k_list:
            print(f"  Top-{k}:")
            for metric_name in ['Precision', 'Recall', 'NDCG', 'Hit_Ratio']:
                key = f'{metric_name}@{k}'
                if key in metrics:
                    print(f"    {metric_name}@{k}: {metrics[key]:.4f}")
        
        print("=" * 60)
        
        return metrics


def evaluate_model(model, data_loader, device, evaluator=None):
    """评估模型性能"""
    if evaluator is None:
        evaluator = RecommenderEvaluator()
    
    model.eval()
    evaluator.reset()
    
    with torch.no_grad():
        for batch in data_loader:
            # 移动数据到设备
            user_ids = batch['user_id'].to(device)
            item_ids = batch['item_id'].to(device)
            ratings = batch['rating'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            images = batch['image'].to(device)
            
            # 前向传播
            predictions = model(user_ids, item_ids, text_input_ids, 
                              text_attention_mask, images)
            
            # 添加到评估器
            evaluator.add_batch(predictions, ratings, user_ids, item_ids)
    
    # 计算指标
    metrics = evaluator.compute_all_metrics()
    
    return metrics, evaluator


if __name__ == "__main__":
    # 测试评估器
    evaluator = RecommenderEvaluator()
    
    # 模拟一些预测结果
    predictions = torch.randn(100)
    targets = torch.randn(100)
    user_ids = torch.randint(0, 10, (100,))
    item_ids = torch.randint(0, 20, (100,))
    
    evaluator.add_batch(predictions, targets, user_ids, item_ids)
    
    # 计算并打印指标
    metrics = evaluator.compute_all_metrics()
    evaluator.print_metrics(metrics)
