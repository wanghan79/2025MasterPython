# %% Imports
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from torch.utils.data.dataloader import DataLoader
# from modeling.ProtBertPPIModel import ProtBertPPIModel
from modeling.EsmPPIModel import EsmPPIModel
from typing import List
import pandas as pd
import esm
from pandas.core.frame import DataFrame
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor)
from torch.nn.utils import clip_grad_norm_
from transformers import EsmTokenizer
from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
import settings
import npe_ppi_logger
from data.VirHostNetDataset import VirHostNetData
# from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger


logger = npe_ppi_logger.get_custom_logger(name=__name__)

def generate_parser():
    parser = ArgumentParser()
    # parser = ProtBertPPIModel.add_model_specific_args(parser)
    # parser._action_groups[1].title = 'Model options'
    parser.add_argument(
        "--deactivate_earlystopping", default=False, type=bool, help="Deactivate early stopping."
    )
    parser.add_argument(
        "--monitor", default="val_loss", type=str, help="Quantity to monitor."
    )
    parser.add_argument(
        "--patience",
        default=10,
        type=int,
        help=(
            "Number of epochs with no improvement "
            "after which training will be stopped."
        ),
    )
    parser.add_argument(
        "--metric_mode",
        default="min",
        type=str,
        help="If we want to min/max the monitored quantity.",
        choices=["auto", "min", "max"],
    )
    parser.add_argument(
        "--nr_frozen_epochs",
        default=1,
        type=int,
        help="Number of epochs we want to keep the encoder model frozen.",
    )
    parser.add_argument(
            "--max_length",
            default=512,
            type=int,
            help="Maximum sequence length.",
        )
    parser.add_argument(
            "--loader_workers",
            default=8,
            type=int,
            help="How many subprocesses to use for data loading. 0 means that \
                the data will be loaded in the main process.",
        )
    parser.add_argument(
        "--learning_rate",
        default=3e-05,
        type=float,
            # options=[1e-05, 3e-05, 5e-05, 1e-04, 3e-04, 5e-04, 1e-03, 3e-03, 5e-03],
            # tunable=True,
        help="Classification head learning rate.",
    )
    
    parser.add_argument(
            "--per_device_predict_batch_size",
            default=8,
            type=int,
            help="The batch size per GPU/TPU core/CPU for test data."
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
        "--adam_epsilon",
        default=1e-08,
        type=float,
            # tunable=True,
            # options=[1e-06, 1e-07, 1e-08, 1e-09],
        help="adam_epsilon"
        )
    parser.add_argument(
        "--weight_decay",
        default=5e-2,
        type=float,
            # options=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
            # tunable=True,
        help="Weight decay for AdamW.",
        )
    parser.add_argument(
        "--dropout_prob",
        default=0.6,
            # tunable=True,
            # options=[0.2,0.3, 0.4, 0.5],
        type=float,
        help="Classification head dropout probability.",
        )
    parser.add_argument(
        "--val_ratio",
        default=0.1,
        type=float,
        help="验证集占训练集的比例 (0.0 ~ 1.0)"
    )
    
    parser.add_argument(
        "--train_csv",
        default=settings.BASE_DATA_DIR + "/train1540.txt",
        type=str,
        help="Path to the file containing the train data.",
    )
    # parser.add_argument(
    #     "--valid_csv",
    #     default=settings.BASE_DATA_DIR + "/val_split.txt",
    #     type=str,
    #     help="Path to the file containing the valid data.",
    # )
    parser.add_argument(
        "--test_csv",
        default=settings.BASE_DATA_DIR + "/test1540.txt",
        type=str,
        help="Path to the file containing the test data.",
    )
    # parser.add_argument(
    #     "--predict_csv",
    #     default=settings.BASE_DATA_DIR + "/generated/sarscov2/ml/predict_omicron_spike_interactions_template.txt",
    #     type=str,
    #     help="Path to the file containing the inferencing data.",
    # )
    # parser.add_argument(
    #     "--k_fold",
    #     default=5,
    #     type=int,
    #     help="Number of folds for cross-validation.",
    # )
    # parser.add_argument(
    #     "--cross_validation_seed",
    #     default=42,
    #     type=int,
    #     help="Random seed for data splitting.",
    # )
    parser.add_argument(
        "--perform_training", 
        default=True, 
        # default=False,
        type=bool, help="Perform training."
    )
    parser.add_argument(
        "--prediction_checkpoint", 
        default=None, 
        # default=settings.BASE_MODELS_DIR + "/final_model_use.ckpt",
        type=str, 
        help="File path of checkpoint to be used for prediction."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        default=True,
        type=bool,
        help="Enable or disable gradient checkpointing which use the cpu memory \
            with the gpu memory to store the model.",
    )

    return parser

def prepare_params():
    
    logger = npe_ppi_logger.get_custom_logger(name=__name__)

    logger.info("Starting parsing arguments...")
    parser = generate_parser()
    params = parser.parse_args()
    logger.info("Finishing parsing arguments.")

    params.encoder_learning_rate = 5e-06
    params.warmup_steps = 1000
    params.max_epochs = 100
    params.min_epochs = 5
    params.local_logger = logger
    params.val_ratio = 0.1  # 默认20%作为验证集
    # params.k_fold = 5  # 默认5折
    # params.cross_validation_seed = 42

    params.label_set = "0,1"

    return params

# def split_train_val(data: pd.DataFrame, val_ratio: float, seed: int = 42) -> tuple:
#     """
#     划分训练集和验证集
#     Args:
#         data: 完整训练集 DataFrame
#         val_ratio: 验证集比例
#         seed: 随机种子
#     Returns:
#         train_df, val_df
#     """
#     # 分层抽样保持类别分布
#     train_df, val_df = train_test_split(
#         data, 
#         test_size=val_ratio,
#         stratify=data['class'],  # 保持正负样本比例
#         random_state=seed
#     )
#     return train_df, val_df

# %% Predict
def main(params):

    if params.perform_training == True:
        logger.info("Starting training.")
    
        model_name = "/mnt/Data6/hjy/STEP/esm2/esm2_t12_35M_UR50D/"
        target_col = 'class'
        seq_col_a = 'sequenceA'
        seq_col_b = 'sequenceB'
        max_len = 512
        batch_size = 1
        seed = 42
        seed_everything(seed)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=settings.BASE_MODELS_DIR,
            filename="interrupted_model",  # 保存文件名
            save_last=True,                # 强制保存中断时的状态
            every_n_epochs=1               # 每个 epoch 保存一次
        )       
        params.callbacks = [checkpoint_callback]
        # 读取完整训练数据
        full_train_data = pd.read_csv(params.train_csv, sep="\t",header=0, encoding="us-ascii")
        
        # 初始化交叉验证
        # kfold = KFold(
        #     n_splits=params.k_fold,
        #     shuffle=True,
        #     random_state=params.cross_validation_seed
        # )

        train_data, val_data = train_test_split(
            full_train_data,
            test_size=params.val_ratio,
            stratify=full_train_data[target_col],  # 保持类别平衡
            random_state=42  # 固定随机种子确保可复现
        )
        train_data = train_data.reset_index(drop=True)  
        val_data = val_data.reset_index(drop=True)

        
        early_stop_callback = EarlyStopping(
            monitor=params.monitor,
            min_delta=0.003,
            patience=params.patience,
            verbose=True,
            mode=params.metric_mode,
            # baseline=0.75,
        )

        params.callbacks = []    
        params.callbacks.append(LearningRateMonitor(logging_interval='step'))

        if params.deactivate_earlystopping == False:
            params.callbacks.append(early_stop_callback)
            print("*********************???",params)
        # Add trainer params from outside: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-in-python-scripts
        trainer = Trainer.from_argparse_args(params)
        
        
        tokenizer = EsmTokenizer.from_pretrained(model_name, do_lower_case=False, local_files_only=True)
        # %% Read set
        # 
        # train_data: DataFrame = pd.read_csv(params.train_csv, sep = "\t", header=0, encoding="us-ascii")
        # val_data: DataFrame = pd.read_csv(params.valid_csv, sep = "\t", header=0, encoding="us-ascii")
        # data = pd.read_csv(params.train_csv, sep="\t", header=0, encoding="utf-16le")
        print("数据列名:", train_data.columns.tolist())
        # print("数据列名:", val_data.columns.tolist())
        
        # train_data, val_data = split_train_val(data, params.val_ratio)
        
        # train_data.to_csv(settings.BASE_DATA_DIR + "/train_split.txt", sep="\t", index=False)
        # val_data.to_csv(settings.BASE_DATA_DIR + "/val_split.txt", sep="\t", index=False)
        
        # train_data = pd.read_csv(settings.BASE_DATA_DIR + "/train_split.txt", sep="\t")
        # val_data = pd.read_csv(settings.BASE_DATA_DIR + "/val_split.txt", sep="\t") 
        
        dataset = VirHostNetData(
            train_data, 
            tokenizer = tokenizer, 
            max_len = max_len, 
            seq_col_a = seq_col_a, 
            seq_col_b = seq_col_b, 
            target_col = target_col
        )
        val_dataset = VirHostNetData(
            val_data,
            tokenizer=tokenizer,
            max_len=max_len,
            seq_col_a = seq_col_a, 
            seq_col_b = seq_col_b, 
            target_col = target_col
        )
        train_loader = DataLoader(
            dataset, 
            batch_size=params.per_device_train_batch_size,
            num_workers=params.loader_workers,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=params.per_device_eval_batch_size,
            num_workers=params.loader_workers
        )
        
        
        # final_metrics = {
        #     'val_acc': [],
        #     'val_f1': [],
        #     'val_auc': []
        # }
        
        # for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
        #     logger.info(f"--- Processing Fold {fold+1}/{params.k_fold} ---")
            
        #     train_subset = Subset(dataset, train_idx)
        #     val_subset = Subset(dataset, val_idx)
            
        #     train_loader = DataLoader(
        #         train_subset,
        #         batch_size=params.per_device_train_batch_size,
        #         num_workers=params.loader_workers,
        #         shuffle=True
        #     )
        #     val_loader = DataLoader(
        #         val_subset,
        #         batch_size=params.per_device_eval_batch_size,
        #         num_workers=params.loader_workers
        #     )
            
        
        model_params = {}
        # TODO: update params from paper
        model_params["encoder_learning_rate"] = 5e-06
        model_params["warmup_steps"] = 200
        model_params["max_epochs"] = 20
        model_params["min_epochs"] = 5
        model = EsmPPIModel(params)
            # callbacks = [pytorch_lightning.callbacks.progress.TQDMProgressBar(refresh_rate=5)]
        trainer = Trainer(
                gpus=[0],
                max_epochs = params.max_epochs, 
                # callbacks=params.callbacks, 
                # checkpoint_callback=checkpoint_callback,
                progress_bar_refresh_rate=1,
                gradient_clip_val=1.0,
                num_sanity_val_steps=0,
                accumulate_grad_batches=8,
                # logger=loggers,
            #logger = npe_ppi_logger.get_mlflow_logger_for_PL(trial.study.study_name)
                # default_root_dir=f"{settings.BASE_MODELS_DIR}/fold_{fold}"
        )
            # trainer.fit(model, train_loader, val_loader)
            # 验证集评估
            # val_results = trainer.validate(model, val_loader)
            
            # 记录指标
            # final_metrics['val_acc'].append(val_results[0]['val_SmartAccuracy'])
            # final_metrics['val_f1'].append(val_results[0]['val_SmartF1Score'])
            # final_metrics['val_auc'].append(val_results[0]['val_AUROC'])

            # 保存当前fold的模型
            # trainer.save_checkpoint(
            #     f"{settings.BASE_MODELS_DIR}/fold_{fold}_model.ckpt"
            # )

        
        train_loader = DataLoader(dataset, batch_size= batch_size, num_workers = 2,pin_memory=True, shuffle=True) # type:ignore
        # val_loader = DataLoader(
        #     val_dataset,
        #     batch_size=batch_size,
        #     num_workers=8
        # )
        # trainer.fit(model, train_loader)
        trainer.fit(model, train_loader, val_loader)
        trainer.save_checkpoint(settings.BASE_MODELS_DIR + "/final_model_use.ckpt")
        
        logger.info("Finishing training.")
        # logger.info(
        #     f"\n=== Final Cross-Validation Results ===\n"
        #     f"Validation Accuracy: {np.mean(final_metrics['val_acc']):.4f} ± {np.std(final_metrics['val_acc']):.4f}\n"
        #     f"Validation F1 Score: {np.mean(final_metrics['val_f1']):.4f} ± {np.std(final_metrics['val_f1']):.4f}\n"
        #     f"Validation AUC: {np.mean(final_metrics['val_auc']):.4f} ± {np.std(final_metrics['val_auc']):.4f}"
        # )

    else:

        df_to_output = pd.read_csv(params.test_csv, sep="\t", header=0, encoding="us-ascii")

        # Load model
        logger.info("Loading model.")

        # model_name = "/mnt/Data6/hjy/STEP/esm2/esm2_t12_35M_UR50D/"
        model: EsmPPIModel = EsmPPIModel.load_from_checkpoint(
            params.prediction_checkpoint, 
            # encoder_name=model_name, 
        )

        batch_size = 1

        # Predict
        logger.info("Loading dataset.")
        print("数据列名:", df_to_output.columns.tolist())
        dataset = VirHostNetData(
            df=df_to_output, 
            tokenizer=model.tokenizer, 
            max_len=512, 
            seq_col_a='sequenceA',
            seq_col_b='sequenceB',
            target_col='class'
        )
            
        logger.info("Predicting.")
        trainer = Trainer(gpus=[0], deterministic=True,max_epochs = 1)
        test_dataloader = DataLoader(dataset, batch_size= batch_size, num_workers = 2, pin_memory=True,shuffle=False) #type: ignore
        test_result = trainer.test(model=model, dataloaders=test_dataloader
            # return_predictions=True
        )
        # for ix in range(len(test_result)): # type:ignore
        #     score = test_result[ix]['probability'][0] # type:ignore
        #     # logger.info("For entry %s, we found a score of %s", ix, score)
        #     df_to_output.at[ix,"score"] = score
        
        df_to_output.to_csv(settings.BASE_DATA_DIR + "predictions.csv", index=False)
            
        # Save results
       

if __name__ == '__main__':
    params = prepare_params()
    main(params)
