from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.base import Callback
import copy
from mlflow.tracking.client import MlflowClient
from optuna import Study
import optuna
import settings

def getXrefByDatabase(line, database):
    fields = line.split('|')

    for field in fields:
        parts = field.split(':')

        db = parts[0]
        value = parts[1].split('(')[0]

        if database == db:
            return value

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.epoch_metrics = []
        self.epochs = []
        self.mlflow_run_id = ""
        self.mlflow_experiment_id = ""

    def on_validation_epoch_end(self, trainer: Trainer, pl_module):
        self.epochs.append(copy.deepcopy(trainer.current_epoch)) # type: ignore
        self.epoch_metrics.append(copy.deepcopy(trainer.callback_metrics))
    
    def on_train_end(self, trainer, pl_module):
        if isinstance(trainer.logger.experiment, MlflowClient):
            self.mlflow_run_id = trainer.logger.run_id  # type: ignore
            self.mlflow_experiment_id = trainer.logger.experiment_id  # type: ignore


def create_or_get_optuna_study(o_fold = None, study_name_prefix = "v-p-ncv", extended_testset = None, storage = settings.OTPUNA_STORAGE) -> Study:
    """
    Create or get optuna study for the specific outer and inner fold

    Args:
        o_fold (int): Outer fold
        study_name_prefix (str): study name prefix
        extended_testset (str): name of extended testset
        storage (str): Optuna storage

    Returns:
        [Study]: Optuna study that was created or loaded 
    """
    if o_fold is not None:
        study_name = study_name_prefix + "-ofold-" + str(o_fold)
    else:
        study_name = study_name_prefix
    
    if extended_testset is not None:
        study_name = study_name + "_" + extended_testset
        
    study: Study = optuna.create_study(
        study_name=study_name,
        storage = storage,
        direction="maximize",  # TODO: move
        pruner=optuna.pruners.HyperbandPruner(), 
        load_if_exists=True
    )

    return study
