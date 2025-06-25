import os
import socket
from utils.SortingHelpFormatter import SortingHelpFormatter
import sys
import esm
from argparse import ArgumentParser
from collections import OrderedDict
from typing import Dict, Tuple

#from plotly.missing_ipywidgets import 
import plotly.express as px
import pytorch_lightning as pl
import torch
from torch.functional import Tensor
import torch.nn as nn

from mlflow.tracking import MlflowClient
from pytorch_lightning.utilities.enums import ModelSummaryMode
from test_tube.argparse_hopt import TTNamespace
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.checkpoint import checkpoint

from torchmetrics import (AUROC, F1Score, ROC, Accuracy, AveragePrecision,
                          MatthewsCorrCoef, Precision, PrecisionRecallCurve,
                          Recall, ConfusionMatrix)
from torchmetrics.collections import MetricCollection
from torchnlp.encoders import LabelEncoder
from torchnlp.utils import collate_tensors
# from transformers import BertModel
from transformers.tokenization_utils_base import BatchEncoding
from transformers.file_utils import TensorType
from transformers import EsmConfig,EsmTokenizer

import settings
from data.PPIDataset import PPIDataset, Dataset
from utils.ProtBertPPIArgParser import ProtBertPPIArgParser
from esm2.models.reference.esm_model import EsmModel

class SmartAccuracy(Accuracy):
    def update(self, preds: Tensor, target: Tensor, *args, **kwargs):
        if preds.dtype == torch.float:
            preds = (preds > 0.5).long()
        return super().update(preds, target, *args, **kwargs)

class SmartPrecision(Precision):
    def update(self, preds: Tensor, target: Tensor, *args, **kwargs):
        if preds.dtype == torch.float:
            preds = (preds > 0.5).long()
        return super().update(preds, target, *args, **kwargs)

class SmartRecall(Recall):
    def update(self, preds: Tensor, target: Tensor, *args, **kwargs):
        if preds.dtype == torch.float:
            preds = (preds > 0.5).long()
        return super().update(preds, target, *args, **kwargs)

class SmartF1Score(F1Score):
    def update(self, preds: Tensor, target: Tensor, *args, **kwargs):
        if preds.dtype == torch.float:
            preds = (preds > 0.5).long()
        return super().update(preds, target, *args, **kwargs)

class SmartConfusionMatrix(ConfusionMatrix):
    def update(self, preds: torch.Tensor, target: torch.Tensor, *args, **kwargs):
        if preds.dtype == torch.float:
            preds = (preds > 0.5).long()
        return super().update(preds, target, *args, **kwargs)

# class SmartMatthewsCorrCoef(MatthewsCorrCoef):
#     def update(self, preds: torch.Tensor, target: torch.Tensor, *args, **kwargs):
#         if preds.dtype == torch.float:
#             preds = (preds > 0.5).long()
#         return super().update(preds, target, *args, **kwargs)


class EsmPPIModel(pl.LightningModule):
    """
    # https://github.com/minimalist-nlp/lightning-text-classification.git
    
    Sample model to show how to use BERT to classify PPIs.
    
    :param params: ArgumentParser containing the hyperparameters.
    """

    def __init__(self, params) -> None:
        """
        Constructor

        TODO: remove dependencies to some of the params, such as train_csv
        TODO: keep in mind, that the construtor is called while loading the checkpoint.

        Args:
            params (dict or TTNamespace): Dictionary or TTNamespace of params. Automatic conversion to TTNamespace if it's a dict.
        """
        super().__init__()

        self.ESM = EsmModel.from_pretrained("/mnt/Data6/hjy/STEP/esm2/esm2_t12_35M_UR50D/", add_pooling_layer=False,use_cache=False)
        
        for layer in self.ESM.encoder.layer:
            layer.gradient_checkpointing = True
        self.ESM.gradient_checkpointing_enable()  # 明确启用ESM的梯度检查点


        # While loading the checkpoint, params are used as dictionary
        if isinstance(params, dict):
            params = TTNamespace(**params)

        self.local_logger = params.local_logger

        # Remove these variables as they cannot be pickled
        # Moreover they can also not be logged with mlflow as deepcopy of these variables do not work
        try:
            # delattr(params,'local_logger')
            delattr(params,'trials')
            delattr(params,'optimize_parallel')
            delattr(params,'optimize_parallel_gpu')
            delattr(params,'optimize_parallel_cpu')
            delattr(params,'optimize_trials_parallel_gpu')
            delattr(params,'generate_trials')
        except AttributeError:
            pass

        if not hasattr(params, 'prog_cwd'):
            setattr(params, 'prog_cwd', os.getcwd())
        if not hasattr(params, 'prog_hostname'):
            setattr(params, 'prog_hostname', socket.gethostname())
        if not hasattr(params, 'prog_arg_string'):
            setattr(params, 'prog_arg_string', sys.argv)

        self.save_hyperparameters(params)

        self.current_val_epoch = 0
        self.current_test_epoch = 0

        self.model_name = "Rostlab/prot_bert_bfd"
        
        self.dataset = PPIDataset()
        # 在 ProtBertPPIModel 的 __init__ 方法中
        

        
        
        # self.dataset.calculate_stat(self.hparams.train_csv)

        self.train_metrics = MetricCollection([
            SmartAccuracy(),  # label 
            SmartPrecision(), # label  
            SmartRecall(), 
            SmartF1Score(),
            AveragePrecision(pos_label=1),
            AUROC(pos_label=1), # logit
            # SmartMatthewsCorrCoef(num_classes=2),
        ], prefix='train_')

        self.valid_metrics = MetricCollection([
            SmartAccuracy(), # label 
            SmartPrecision(), # label 
            SmartRecall(), # label
            SmartF1Score(), # label 
            AveragePrecision(pos_label=1), # logit 
            SmartConfusionMatrix(num_classes=2,), # label 
            PrecisionRecallCurve(pos_label=1), # logit 
            AUROC(pos_label=1,average=None), # logit 
            ROC(pos_label=1), # logit 
            # SmartMatthewsCorrCoef(num_classes=2), # label 
        ], prefix='val_')

        self.test_metrics = self.valid_metrics.clone(prefix="test_")

        # build model
        self.__build_model()

        # Loss criterion initialization.
        self.__build_loss()

        
        # freeze encoder if user mants it
        if self.hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False # don't need to call unfreeze_encoder, as it is always in unfrozen state.

    def __build_model(self) -> None:
        """ Init BERT model + tokenizer + classification head."""
        
        config = EsmConfig.from_pretrained('/mnt/Data6/hjy/STEP/esm2/esm2_t12_35M_UR50D/')
        config.gradient_checkpointing = True
        
        self.original_esm_dim = 480  # ESM-2 12M模型的隐藏层维度
        self.adapted_dim = 512     

        self.dim_adapter = nn.Sequential(
            nn.Linear(self.original_esm_dim , self.adapted_dim ),  # 1920->2048
            nn.GELU(),
            nn.LayerNorm(self.adapted_dim )
        )

        # Tokenizer
        self.tokenizer = EsmTokenizer.from_pretrained("/mnt/Data6/hjy/STEP/esm2/esm2_t12_35M_UR50D/",add_token_type_ids=False)
        # Label Encoder
        self.label_encoder = LabelEncoder(
            self.hparams.label_set.split(","), reserved_labels=[]
        )
        # self.label_encoder.unknown_index = None

        self.classification_head = nn.Sequential(OrderedDict([
            ("dropout1", nn.Dropout(self.hparams.dropout_prob)),
            ("dense1", nn.Linear(self.adapted_dim , 1)),  # 2048->128
            # ("dropout2", nn.Dropout(0.2)),
            # ("dense2", nn.Linear(int(self.adapted_dim * 4 / 16), int(self.adapted_dim * 4 / (16*16)))),  # 128->8
            # ("dropout3", nn.Dropout(0.2)),
            # ("dense3", nn.Linear(int(self.adapted_dim * 4 / (16*16)), 1)),  # 8->1
        ]))

        self.sigmoid = nn.Sigmoid()

    def classifier(self, model_out_A, model_out_B):
        x = model_out_A * model_out_B
        result = self.classification_head(x)
        result = result.view(-1)
        return {"logits": result}

    def __build_loss(self) -> None:
        """ Initializes the loss function/s. """
        self._loss_bce_with_integrated_sigmoid = nn.BCEWithLogitsLoss()

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            if self.global_rank == 0:
                self.local_logger.info(f"-- Encoder model fine-tuning")
            for param in self.ESM.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        if self.global_rank == 0:
            self.local_logger.info(f"-- Freezing encoder model")
        for param in self.ESM.parameters():
            param.requires_grad = False
        self._frozen = True

    def pool_strategy(self, features):
        """修改后的池化策略，包含维度适配"""
        token_embeddings = features['token_embeddings']  # [batch, seq_len, 480]
        attention_mask = features['attention_mask']      # [batch, seq_len]
    
        # 四种池化方式
        pool_vectors = []
    
        # 1. CLS Token池化
        pool_vectors.append(features['cls_token_embeddings'])  # [batch, 640]
    
        # 2. Max Pooling
        # input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # token_embeddings[input_mask_expanded == 0] = -1e9
        # max_pool = torch.max(token_embeddings, 1)[0]  # [batch, 640]
        # pool_vectors.append(max_pool)
    
        # 3. Mean Pooling
        # sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        # sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
        # mean_pool = sum_embeddings / sum_mask  # [batch, 640]
        # pool_vectors.append(mean_pool)
    
        # 4. Sqrt Mean Pooling
        # sqrt_mean_pool = sum_embeddings / torch.sqrt(sum_mask)  # [batch, 640]
        # pool_vectors.append(sqrt_mean_pool)
    
        # 拼接所有池化结果 [batch, 640*4=2560]
        pooled_output = torch.cat(pool_vectors, dim=1)
    
        # 关键修改3：通过适配层调整维度 [batch, 2560] -> [batch, 4096]
        adapted_output = self.dim_adapter(pooled_output)
    
        return adapted_output
                      

    def forward(self, input_ids, attention_mask):
        """
        Usual pytorch forward function.
        
        TODO: Refactor for retrieving input for both sequences

        Args:
            input_ids ([type]): token ids of input text sequences
            # token_type_ids ([type]): token type idss of input text sequences
            attention_mask ([type]): attention mask of input test sequences.

        Returns:
            [List]: List of pooled vectors
        """
        outputs = self.ESM(input_ids=input_ids, attention_mask=attention_mask)
        word_embeddings = outputs.last_hidden_state

        pooling = self.pool_strategy({
            "token_embeddings": word_embeddings,
            "cls_token_embeddings": word_embeddings[:, 0],
            "attention_mask": attention_mask,
        })

        return pooling

    def loss_bce_with_integrated_sigmoid(self, predictions: dict, targets: dict) -> torch.Tensor:
        """
        Computes Loss value according to a loss function.
        :param predictions: model specific output. Must contain a key 'logits' with
            a tensor [batch_size x 1] with model predictions
        :param labels: Label values [batch_size]
        Returns:
            torch.tensor with loss value.
        """
        return self._loss_bce_with_integrated_sigmoid(predictions["logits"], targets["labels"].float())

    def prepare_sample_without_target(self, sample: list):
        collated_sample = collate_tensors(sample) #type: ignore
        inputs_A, inputs_B, _ = self.prepare_sample(sample, prepare_target = False)

        return inputs_A, inputs_B, collated_sample

    def prepare_sample(self, sample: list, prepare_target: bool = True) -> Tuple[BatchEncoding, BatchEncoding, Dict]:
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.
        
        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
        collated_sample = collate_tensors(sample) #type: ignore
        
        inputs_A = self.tokenizer.batch_encode_plus(collated_sample["seqA"],
                                                  add_special_tokens=True,
                                                  padding=True,
                                                  truncation=True,
                                                  return_tensors=TensorType.PYTORCH,
                                                  max_length=self.hparams.max_length,
                                                  return_token_type_ids=False)
                                                  

        inputs_B = self.tokenizer.batch_encode_plus(collated_sample["seqB"],
                                                  add_special_tokens=True,
                                                  padding=True,
                                                  truncation=True,
                                                  return_tensors=TensorType.PYTORCH,
                                                  max_length=self.hparams.max_length,
                                                  return_token_type_ids=False)

        # print(inputs_A)  

        if "token_type_ids" in inputs_A:
            del inputs_A["token_type_ids"]
        if "token_type_ids" in inputs_B:
            del inputs_B["token_type_ids"]

        if not prepare_target:
            return inputs_A, inputs_B, {}

        # Prepare target:
        try:
            targets = {"labels": self.label_encoder.batch_encode(collated_sample["label"])}
            # TODO: Return also the protein ids and ncbi gene id
            return inputs_A, inputs_B, targets
        except RuntimeError:
            if self.global_rank == 0:
                self.local_logger.error("Label encoder found an unknown label: {}", collated_sample["label"])
            raise Exception("Label encoder found an unknown label.")

    def on_train_start(self) -> None:
        super().on_train_start()

        if self.global_rank == 0 and isinstance(self.logger.experiment, MlflowClient):
            self.mlflow : MlflowClient = self.logger.experiment
        
            # Save model description to mlflow artifacts
            # self.mlflow.log_text(self.logger.run_id, str(ModelSummary(self, mode=ModelSummaryMode.FULL)), "./model/model_summary.txt")
            self.mlflow.log_text(self.logger.run_id, str(self), "./model/model_summary_with_params.txt")
            self.local_logger.info("Training started, check out run: %s", settings.MLFLOW_TRACKING_URI + "/#/experiments/" + self.logger.experiment_id + "/runs/" + self.logger.run_id)

    def __single_step(self, batch):
        inputs_A, inputs_B, targets = batch
        model_out_A = self.forward(**inputs_A)
        model_out_B = self.forward(**inputs_B)
        classifier_output = self.classifier(model_out_A, model_out_B)

        loss = self.loss_bce_with_integrated_sigmoid(classifier_output, targets)

        trues = targets["labels"]
        preds = classifier_output["logits"]
        preds = self.sigmoid(preds)

        return (loss, trues, preds)

    
    def training_step(self, batch, batch_idx):
        inputs_A, inputs_B, targets = batch
    
        # Forward pass
        model_out_A = self.forward(**inputs_A)  # [batch, 4096]
        model_out_B = self.forward(**inputs_B)  # [batch, 4096]
        # logits = self.classifier(model_out_A, model_out_B)
        classifier_output = self.classifier(model_out_A, model_out_B)  # 正确获取分类器输出字典

        # 提取logits张量
        logits = classifier_output["logits"]
    
        train_loss, trues, preds = self.__single_step(batch)

        self.train_metrics.update(preds, trues.long())

        self.log('train_loss', train_loss, on_step=False, on_epoch=True)
        
        # Loss计算
        loss = self.loss_bce_with_integrated_sigmoid({"logits": logits}, targets)
    
        
        
        # 关键修改4：梯度监控
        if batch_idx % 100 == 0:
            # grads = {
            #     "ESM_emb": self.ESM.embeddings.word_embeddings.weight.grad.norm().item(),
            #     "adapter": self.dim_adapter[0].weight.grad.norm().item(),
            #     "head1": self.classification_head[1].weight.grad.norm().item()
            # }
            grads = {}
            # 检查ESM词嵌入梯度是否存在
            if self.ESM.embeddings.word_embeddings.weight.grad is not None:
                grads["ESM_emb"] = self.ESM.embeddings.word_embeddings.weight.grad.norm().item()
            else:
                grads["ESM_emb"] = 0.0  # 冻结时设为0或跳过
    
            # 检查适配层梯度
            if self.dim_adapter[0].weight.grad is not None:
                grads["adapter"] = self.dim_adapter[0].weight.grad.norm().item()
            else:
                grads["adapter"] = 0.0
    
            # 检查分类头梯度
            if self.classification_head[1].weight.grad is not None:
                grads["head1"] = self.classification_head[1].weight.grad.norm().item()
            else:
                grads["head1"] = 0.0
            self.log_dict({f"grad_{k}": v for k,v in grads.items()}, prog_bar=True)
            print(f"\n梯度监控 (batch={batch_idx}):")
            print(f"  ESM词嵌入梯度范数: {grads['ESM_emb']:.3e} (理想范围:1e-5~1e-3)")
            print(f"  适配层梯度范数: {grads['adapter']:.3e} (理想范围:1e-4~1e-2)")
            print(f"  分类头第一层梯度范数: {grads['head1']:.3e} (理想范围:1e-3~1e-1)")
        
        torch.cuda.empty_cache()
        return loss

    def training_epoch_end(self, outputs: list) -> None:
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.
        """
        result = self.train_metrics.compute()
        self.train_metrics.reset()
        
        result.pop('train_ROC', None)
        result.pop('train_PrecisionRecallCurve', None)
        self.log_dict(result, on_epoch=True)

        if self.global_rank == 0:
            self.local_logger.info("Training epoch %s finished", self.current_epoch)
            if isinstance(self.logger.experiment, MlflowClient):
                self.local_logger.info("Check out run: %s", settings.MLFLOW_TRACKING_URI + "/#/experiments/" + self.logger.experiment_id + "/runs/" + self.logger.run_id)

        # check for unfreezing encoder
        if self.current_epoch + 1 >= self.hparams.nr_frozen_epochs:
            self.unfreeze_encoder()

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        val_loss, trues, preds = self.__single_step(batch)

    # 将预测值转换为概率（如使用BCEWithLogitsLoss需添加sigmoid）
    # 根据 preds 的维度选择 argmax 的维度
        # self.valid_metrics.update((probs > 0.5).long(), trues)  # 需要类别标签
        self.valid_metrics.update(preds, trues.long())
        output = OrderedDict({
            'val_loss': val_loss,
        })

        return output

    def validation_epoch_end(self, outputs: list) -> None:
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.
        """
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        result = self.valid_metrics.compute()
        self.valid_metrics.reset()

        self.valid_metrics.reset()
        # self.auc_metric.reset()

        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)

        # self.log_roc_graph(result, self.valid_metrics.prefix)
        # self.log_prc_graph(result, self.valid_metrics.prefix)

        # do not log ROC and PRC
        result.pop(self.valid_metrics.prefix + 'ROC', None)
        result.pop(self.valid_metrics.prefix + 'PrecisionRecallCurve', None)
        result.pop(self.valid_metrics.prefix + 'ConfusionMatrix', torch.Tensor([[-1,-1],[-1,-1]]))
        self.log_dict(result, on_epoch=True)
        
        self.current_val_epoch += 1

    def test_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        test_loss, trues, preds = self.__single_step(batch)

        self.test_metrics.update(preds, trues.long())
        
        output = OrderedDict({
            'test_loss': test_loss,
        })

        return output

    def test_epoch_end(self, outputs: list) -> None:
        """ Function that takes as input a list of dictionaries returned by the test_step
        function and measures the model performance accross the entire validation set.
        """
        test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        result = self.test_metrics.compute()
        self.test_metrics.reset()

        self.log('test_loss', test_loss, on_epoch=True, prog_bar=True)
        
        #self.log_roc_graph(result, self.test_metrics.prefix)
        #self.log_prc_graph(result, self.test_metrics.prefix)

        # do not log ROC and PRC
        result.pop(self.test_metrics.prefix + 'ROC', None)
        result.pop(self.test_metrics.prefix + 'PrecisionRecallCurve', None)
        result.pop(self.test_metrics.prefix + 'ConfusionMatrix', torch.Tensor([[-1,-1],[-1,-1]]))
        self.log_dict(result, on_epoch=True)

        self.current_test_epoch += 1

    def predict_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        inputs_A, inputs_B, collated_samples = batch
        model_out_A = self.forward(**inputs_A)
        model_out_B = self.forward(**inputs_B)
        classifier_output = self.classifier(model_out_A, model_out_B)

        preds = classifier_output["logits"]
        preds = self.sigmoid(preds)

        collated_samples["probability"] = [p.item() for p in preds]

        return collated_samples

    def predict(self, sample: dict) -> dict:
        """
        Predict function

        Args:
            sample (dict): dictionary with two sequences "seqA" and "seqB" we want to predict interaction for.

        Returns:
            dict: Dictionary with the sequences and the predicted probability.
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            model_inputA, model_inputB, _ = self.prepare_sample([sample], prepare_target=False)
            model_out_A = self.forward(**model_inputA)
            model_out_B = self.forward(**model_inputB)
            classifier_output = self.classifier(model_out_A, model_out_B)
            logits = classifier_output["logits"]
            preds = self.sigmoid(logits)

            sample["probability"] = preds[0].item()

        return sample

    @property
    def num_training_steps(self) -> int:
        """
        Total training steps inferred from datamodule and devices.
        
        https://github.com/PyTorchLightning/pytorch-lightning/issues/5449#issuecomment-774265729
        """
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    def lr_lambda(self, current_step) -> float:
        """
        Calculate learning rate for current step according to the total number of training steps

        Args:
            current_step (int): Current step number

        Returns:
            [float]: learning rate lambda (how much the rate should be changed.)
        """
        warmup_steps = 1000  # 增加到1000步预热
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(self.num_training_steps - current_step) / float(max(1, self.num_training_steps - warmup_steps)))

    def configure_optimizers(self):
        """
        Confiugre the optimizers and schedulears.

        It also sets different learning rates for different parameter groups. 
        """
        # no_decay_params = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": self.ESM.parameters(),
                "lr": 1e-5,  # ESM层使用更小的学习率
                "weight_decay": 0.01
            },
            {
                "params": self.dim_adapter.parameters(),
                "lr": 3e-5,
                "weight_decay": 0.01
            },
            {
                "params": self.classification_head.parameters(),
                "lr": 5e-5,  # 分类头使用较大学习率
                "weight_decay": 0.01
            },
        ]

        # parameters = optimizer_grouped_parameters
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            eps=1e-8,
            betas=(0.9, 0.999)
        )

        # scheduler = LambdaLR(optimizer, self.lr_lambda)
        scheduler = {
            'scheduler': LambdaLR(optimizer, lr_lambda=self.lr_lambda),
            'interval': 'step',
            'frequency': 1,
            'name': 'learning_rate'
        }

        return [optimizer], [scheduler]

    def __retrieve_dataset(self, train=False, val=False, test=False, predict=False) -> Dataset:
        """ Retrieves task specific dataset """
        if train:
            return self.dataset.load_dataset(self.hparams.train_csv)
        elif val:
            return self.dataset.load_dataset(self.hparams.valid_csv)
        elif test:
            return self.dataset.load_dataset(self.hparams.test_csv)
        elif predict:
            return self.dataset.load_predict_dataset(self.hparams.predict_csv)
        else:
            raise Exception('Incorrect dataset split')
    
    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        self._train_dataset = self.__retrieve_dataset(train=True)
        return DataLoader(
            dataset=self._train_dataset,
            sampler=RandomSampler(self._train_dataset),
            batch_size=self.hparams.per_device_train_batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Function that loads the validation set."""
        return None
        # self._dev_dataset = self.__retrieve_dataset(val=True)
        # return DataLoader(
        #     dataset=self._dev_dataset,
        #     batch_size=self.hparams.per_device_eval_batch_size,
        #     collate_fn=self.prepare_sample,
        #     num_workers=self.hparams.loader_workers,
        # )

    def test_dataloader(self) -> DataLoader:
        """Function that loads the validation set. """
        self._test_dataset = self.__retrieve_dataset(test=True)
        return DataLoader(
            dataset=self._test_dataset,
            batch_size=self.hparams.per_device_test_batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        """Function that loads the validation set."""
        self._predict_dataset = self.__retrieve_dataset(predict=True)
        return DataLoader(
            dataset=self._predict_dataset,
            batch_size=self.hparams.per_device_predict_batch_size,
            collate_fn=self.prepare_sample_without_target,
            num_workers=self.hparams.loader_workers,
        )

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ProtBertPPIArgParser:
        """ Parser for Estimator specific arguments/hyperparameters. 
        :param parser: HyperOptArgumentParser obj
        Returns:
            - updated parser
        """

        parser = ProtBertPPIArgParser(
            strategy="random_search",
            description="Minimalist ProtBERT Classifier",
            add_help=False,
            parents=[parent_parser],
            formatter_class=SortingHelpFormatter
        )
        parser.opt_list(
            "--adam_epsilon",
            default=1e-08,
            type=float,
            tunable=True,
            options=[1e-06, 1e-07, 1e-08, 1e-09],
            help="adam_epsilon"
        )
        parser.add_argument(
            "--nb_trials",
            default=24,
            type=int,
            help="Number of trials to run"
        )
        parser.add_argument(
            "--per_device_train_batch_size",
            default=8,
            type=int,
            help="Batch size to be used for training data."
        )
        parser.add_argument(
            "--per_device_eval_batch_size",
            default=8,
            type=int,
            help="The batch size per GPU/TPU core/CPU for validation data."
        )
        parser.add_argument(
            "--per_device_test_batch_size",
            default=8,
            type=int,
            help="The batch size per GPU/TPU core/CPU for test data."
        )
        parser.add_argument(
            "--per_device_predict_batch_size",
            default=8,
            type=int,
            help="The batch size per GPU/TPU core/CPU for test data."
        )
        parser.add_argument(
            "--max_length",
            default=512,
            type=int,
            help="Maximum sequence length.",
        )
        parser.add_argument(
            "--encoder_learning_rate",
            default=5e-06,
            type=float,
            help="Encoder specific learning rate.",
        )
        parser.opt_list(
            "--learning_rate",
            default=3e-05,
            type=float,
            options=[1e-05, 3e-05, 5e-05, 1e-04, 3e-04, 5e-04, 1e-03, 3e-03, 5e-03],
            tunable=True,
            help="Classification head learning rate.",
        )
        parser.opt_list(
            "--weight_decay",
            default=1e-2,
            type=float,
            options=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
            tunable=True,
            help="Weight decay for AdamW.",
        )
        parser.add_argument(
            "--warmup_steps",
            default=200,
            type=int,
            help="Warm up steps for learning rate schedular.",
        )
        parser.opt_list(
            "--dropout_prob",
            default=0.5,
            tunable=True,
            options=[0.2,0.3, 0.4, 0.5],
            type=float,
            help="Classification head dropout probability.",
        )
        parser.opt_list(
            "--nr_frozen_epochs",
            default=1,
            type=int,
            help="Number of epochs we want to keep the encoder model frozen.",
            tunable=True,
            options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )
        parser.add_argument(
            "--gradient_checkpointing",
            default=True,
            type=bool,
            help="Enable or disable gradient checkpointing which use the cpu memory \
                with the gpu memory to store the model.",
        )
        # Data Args:
        parser2 = parser.add_argument_group("Data options")
        parser2.add_argument(
            "--label_set",
            default="1,0",
            type=str,
            help="Classification labels set.",
        )
        parser2.add_argument(
            "--train_csv",
            default=settings.BASE_DATA_DIR + "/generated/vp1/ml/train.txt",
            type=str,
            help="Path to the file containing the train data.",
        )
        parser2.add_argument(
            "--valid_csv",
            default=settings.BASE_DATA_DIR + "/generated/vp1/ml/valid.txt",
            type=str,
            help="Path to the file containing the valid data.",
        )
        parser2.add_argument(
            "--test_csv",
            default=settings.BASE_DATA_DIR + "/generated/vp1/ml/test.txt",
            type=str,
            help="Path to the file containing the test data.",
        )
        parser2.add_argument(
            "--predict_csv",
            default=settings.BASE_DATA_DIR + "/generated/vp1/ml/predict_vp1.txt",
            type=str,
            help="Path to the file containing the inferencing data.",
        )
        parser2.add_argument(
            "--loader_workers",
            default=8,
            type=int,
            help="How many subprocesses to use for data loading. 0 means that \
                the data will be loaded in the main process.",
        )

        return parser
