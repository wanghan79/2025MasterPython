import warnings
warnings.filterwarnings('ignore')
from load_model import load_model
def train(base_model,data_path,cfg_path):
    model = load_model(base_model)#这里直接加载yolo模型训练
    model.train(data=data_path,#配置数据集的配置文件
                cfg=cfg_path#数据增强、优化器、周期、模型保存、超参数等配置文件
                )
    
if __name__ == '__main__':
    train(r"yolo11n.pt",
          r'..\yolo_hanzi_dataset\dataset.yaml',
          r'ultralytics\cfg\default.yaml')
    