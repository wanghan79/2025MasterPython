import os
import logging

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# 目录配置
DATA_DIR = "data/pdb"
OUTPUT_DIR = "output/samples"
LOG_DIR = "logs/pdb_logs"
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
PDB_OUT_DIR = os.path.join(OUTPUT_DIR, "pdbs")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
NPY_OUT_DIR = os.path.join(OUTPUT_DIR, "npy")

# 模型超参数
EMBED_DIM = 128
HIDDEN_DIM = 256
LATENT_DIM = 64
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
EPOCHS = 50
VAL_SPLIT = 0.2

# 创建必要的目录
for path in [LOG_DIR, MODELS_DIR, PDB_OUT_DIR, PLOT_DIR, NPY_OUT_DIR]:
    os.makedirs(path, exist_ok=True)

# 日志配置
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'train.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
