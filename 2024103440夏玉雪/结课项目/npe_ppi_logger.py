import logging
import settings
from mlflow.projects import utils
from mlflow.tracking.context import registry
from mlflow.utils.mlflow_tags import (MLFLOW_GIT_COMMIT, MLFLOW_PARENT_RUN_ID,
                                      MLFLOW_SOURCE_NAME)
from pytorch_lightning.loggers import MLFlowLogger
import optuna

def get_custom_logger(name: str, level=settings.LOGGING_LEVEL):
    # create formatter
    # formatter = logging.Formatter(fmt='%(asctime)s %(filename)s %(module)s: %(levelname)8s %(message)s')
    # date_format = '%m-%d %H:%M:%S'
    # formatter = logging.Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s', datefmt = date_format)
    formatter = logging.Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d :: %(funcName)20s()} -  %(levelname)s - %(message)s')

    # create console handler and set level to debug
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    #logger.handlers = []
    logger.addHandler(handler)

    # modify some of the loggers
    modifiy_loggers()
    
    return logger


def modfiy_available_logger(logger: logging.Logger, level=settings.EXTERNAL_LOGGING_LEVEL):
    # create formatter
    # formatter = logging.Formatter(fmt='%(asctime)s %(filename)s %(module)s: %(levelname)8s %(message)s')
    # date_format = '%m-%d %H:%M:%S'
    # formatter = logging.Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s', datefmt = date_format)
    formatter = logging.Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d :: %(funcName)20s()} -  %(levelname)s - %(message)s')

    # create console handler and set level to debug
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # create logger
    if level is not None:
       logger.setLevel(level)
    logger.propagate = False
    logger.handlers = []
    logger.addHandler(handler)

    return logger

def get_mlflow_logger_for_PL(exp_name = "Fine-tuning exp") -> MLFlowLogger:
    """
    Prepare MLFlow logger for pytorch lightning (PL)

    Args:
        exp_name (str, optional): Name of the experiment. Defaults to "Fine-tuning exp".

    Returns:
        MLFlowLogger: MLFlow logger object of PL
    """
    tags  = registry.resolve_tags(tags=None)
    # tags["git_url"] = str(utils._get_git_repo_url(settings.home_base_dir)).replace(".git", "")
    # tags["git_url"] += "/-/blob/" + tags[MLFLOW_GIT_COMMIT] 
    # tags["git_url"] += tags[MLFLOW_SOURCE_NAME].replace(settings.home_base_dir,"")
    mlflow_logger = MLFlowLogger(
        experiment_name=exp_name,
        tracking_uri=settings.MLFLOW_TRACKING_URI,
        tags = tags
        #    artifact_location=settings.MLFLOW_ARTIFACTS_DIR
    )
    return mlflow_logger

def modifiy_loggers():
    """
    Prepare logger of libraries

    Currently it modifies logger of following libraries:
     - pytorch_lightning
     - optuna

    Returns:
        [type]: [description]
    """
    logger = logging.getLogger("pytorch_lightning")
    modfiy_available_logger(logger)
    modfiy_available_logger(logging.getLogger(optuna.__name__))
    ## logging.basicConfig()
    ## setLevel(logging.DEBUG)
    # logger = logging.getLogger('sqlalchemy.engine')
    # modfiy_available_logger(logger, level = logging.INFO )

