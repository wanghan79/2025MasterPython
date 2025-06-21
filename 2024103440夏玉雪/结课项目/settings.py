import logging
import platform
import socket

# print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
# Defaults

GIT_REPO_NAME="STEP"

LOGGING_LEVEL = logging.INFO
EXTERNAL_LOGGING_LEVEL = logging.ERROR

home_base_dir = "/mnt/Data6/hjy/" + GIT_REPO_NAME
BASE_DATA_DIR = home_base_dir + "/data"
TEST_TUBE_LOG_PATH = home_base_dir + "/test_tube/logs"
BASE_MODELS_DIR = home_base_dir + "/models"
BASE_PRETRAINED_MODEL_DIR = BASE_MODELS_DIR + "/pre-trained-models"

MLFLOW_SERVER_PORT="6000"
MLFLOW_SERVER_HOST = "localhost"
MLFLOW_BACKEND_STORE_URI="sqlite:///" + home_base_dir + "/mlflow_runs/database/mlflow.db"
MLFLOW_ARTIFACTS_DIR = "file:///" + home_base_dir + "/mlflow_runs/artifacts/"

OTPUNA_STORAGE="sqlite:///" + home_base_dir + "/example.db"

# Defaults
TMP_DIR = home_base_dir + "/tmp"

MLFLOW_TRACKING_URI = "http://" + MLFLOW_SERVER_HOST + ":" + MLFLOW_SERVER_PORT
