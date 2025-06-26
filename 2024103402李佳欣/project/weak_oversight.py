import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os
import json
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path_mapper = "/input/dataset"

    class Domain:
        name = "academic-thesis-comments"
        sbert_mapper = "uer/sbert-base-chinese-nli"
        cat_threshold = 0.75
        aspect_category_mapper = [
            "理论", "能力", "价值",
            "规范", "相关性", "立场"
        ]
        with open("/input/dataset/data.json", encoding="utf-8") as f:
            seed_data = json.load(f)
        aspect_seed_mapper = {k: v["正面"] + v["负面"] for k, v in seed_data.items()}

    domain = Domain()


def load_training_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    sentences = []
    train_data = {}
    ids = []
    for id_, record in data.items():
        text = record['text']
        if not isinstance(text, str):
            text = str(text)
        label = record['label']
        sentences.append(text)
        ids.append(id_)
        train_data[text] = {"label": label, "id": id_}
    return sentences, ids, train_data


def load_evaluate_data(file_path):
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    records = {}
    for id_, item in data.items():
        text = item["text"].strip()
        aspects = item.get("aspect", [])
        indices = [Config.domain.aspect_category_mapper.index(a) for a in aspects if
                   a in Config.domain.aspect_category_mapper]
        records[text] = indices
    test_sentences = list(records.keys())
    test_labels = list(records.values())
    return test_sentences, test_labels


class Labeler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = SentenceTransformer(cfg.domain.sbert_mapper, device=cfg.device)
        self.cat_threshold = cfg.domain.cat_threshold
        self.root_path = cfg.path_mapper
        self.categories = cfg.domain.aspect_category_mapper
        self.category_sentences = cfg.domain.aspect_seed_mapper

    def __call__(self, evaluate=True, load=False):
        seeds = {cat: list(self.category_sentences[cat]) for cat in self.categories}
        self.sentences, self.ids, self.train_data = load_training_data(f'{self.root_path}/train_aspect.json')
        seed_embeddings, embeddings = self.__sbert_embedder(load, seeds)
        cosine_scores = []
        for seed_embedding in seed_embeddings:
            total_tensor = torch.cat((seed_embedding, torch.mean(seed_embedding, dim=0).unsqueeze(0)))
            cosine_scores.append(torch.max(util.cos_sim(total_tensor, embeddings), dim=0)[0].unsqueeze(dim=-1))
        cosine_category_scores = torch.cat(cosine_scores, 1)
        for i, sentence in enumerate(self.sentences):
            cat_indices = torch.where(cosine_category_scores[i] >= self.cat_threshold)[0]
            predicted_labels = [self.categories[c.item()] for c in cat_indices]
            id_ = self.ids[i]
            if sentence in self.train_data:
                self.train_data[sentence]["label"] = predicted_labels
                self.train_data[sentence]["text"] = sentence
        output_path = os.path.join("/working", "train-labeled.json")
        with open(output_path, 'w', encoding='utf-8') as jf:
            json.dump(self.train_data, jf, ensure_ascii=False, indent=2)
        print(f"已保存弱监督标注后的训练集到 {output_path}")
        if evaluate:
            self.evaluate(seeds)

    def evaluate(self, seeds):
        test_sentences, test_labels = load_evaluate_data(f'{self.root_path}/test_aspect.json')
        test_embeddings = self.model.encode(test_sentences, convert_to_tensor=True, show_progress_bar=True)
        seed_embeddings = [self.model.encode(seed, convert_to_tensor=True) for seed in seeds.values()]
        cosine_test_scores = []
        for seed_embedding in seed_embeddings:
            total_tensor = torch.cat((seed_embedding, torch.mean(seed_embedding, dim=0).unsqueeze(0)))
            cosine_test_scores.append(
                torch.max(util.cos_sim(total_tensor, test_embeddings), dim=0)[0].unsqueeze(dim=-1))
        cosine_category_scores = torch.cat(cosine_test_scores, 1)
        y_pred = []
        for i in range(len(test_sentences)):
            pred_labels = []
            cat_indices = torch.where(cosine_category_scores[i] >= self.cat_threshold)[0]
            for c in cat_indices:
                pred_labels.append(c.item())
            y_pred.append(pred_labels)
        mlb = MultiLabelBinarizer(classes=list(range(len(self.categories))))
        y_true_bin = mlb.fit_transform(test_labels)
        y_pred_bin = mlb.transform(y_pred)
        print("\n多标签（方面）评估报告：")
        report = classification_report(
            y_true_bin, y_pred_bin,
            target_names=self.categories,
            digits=4,
            output_dict=True
        )
        print(classification_report(y_true_bin, y_pred_bin, target_names=self.categories, digits=4))
        metrics = ['precision', 'recall', 'f1-score']
        data = {m: [report[c][m] for c in self.categories] for m in metrics}
        df_metrics = pd.DataFrame(data, index=self.categories)
        plt.figure(figsize=(12, 6))
        df_metrics.plot(kind='bar')
        plt.title('各类别 Precision/Recall/F1-score')
        plt.ylabel('分数')
        plt.xlabel('类别')
        plt.ylim(0, 1)
        plt.xticks(rotation=30)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.root_path, 'aspect_metrics_bar.png'))
        plt.show()
        pred_counts = np.sum(y_pred_bin, axis=0)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=self.categories, y=pred_counts)
        plt.title('各类别预测数量分布')
        plt.ylabel('预测数量')
        plt.xlabel('类别')
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(os.path.join(self.root_path, 'aspect_pred_count.png'))
        plt.show()

        thresholds = np.arange(0.1, 0.91, 0.05)
        f1s, precisions, recalls = [], [], []
        for th in thresholds:
            preds = (cosine_category_scores[:, 1] > th).astype(int)
            f1s.append(f1_score(y_true_bin, preds, average='micro', zero_division=0))
            precisions.append(precision_score(y_true_bin, preds, average='micro', zero_division=0))
            recalls.append(recall_score(y_true_bin, preds, average='micro', zero_division=0))

        plt.figure(figsize=(8,5))
        plt.plot(thresholds, f1s, label='F1')
        plt.plot(thresholds, precisions, label='Precision')
        plt.plot(thresholds, recalls, label='Recall')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('阈值-性能曲线')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.root_path, 'threshold_curve.png'))
        plt.show()

    def __sbert_embedder(self, load, seeds):
        seed_embeddings = [self.model.encode(seed, convert_to_tensor=True) for seed in list(seeds.values())]
        cache_path = os.path.join("/working", "sbert_train_embeddings_aspect.pickle")
        if load and os.path.isfile(cache_path):
            print(f'Loading embeddings from {cache_path}')
            embeddings = torch.load(cache_path)
        else:
            if not all(isinstance(text, str) for text in self.sentences):
                raise ValueError("self.sentences 中包含非字符串类型的数据")
            embeddings = self.model.encode(self.sentences, convert_to_tensor=True, show_progress_bar=True)
            print(f'Saving embeddings to {cache_path}')
            torch.save(embeddings, cache_path)
        return seed_embeddings, embeddings


if __name__ == '__main__':
    cfg = Config()
    labeler = Labeler(cfg)
    labeler(evaluate=True, load=True)
