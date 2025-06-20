from ultralytics import YOLO
import torch
import os
from torch import nn
def load_model(model_path: str, device: str = None):#这是返回yolo对象的模型
    try:
        # 检查模型路径是否存在
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"模型文件错误，请修改加载路径: {model_path}")
            
        # 自动选择设备为GPU
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        # 验证设备是否可用
        if device == 'cuda' and not torch.cuda.is_available():
            print("警告: CUDA不可用，将使用CPU")
            device = 'cpu'
            
        # 加载模型
        model = YOLO(model_path).to(device)
        print(f"成功加载模型到 {device.upper()}")
        return model
        
    except Exception as e:
        raise RuntimeError(f"加载模型失败: {str(e)}")
    



# --------------------------------------------------
# load_pytorch_module (返回 torch.nn.Module)
# --------------------------------------------------
def load_pytorch_module(model_path: str, device: str = None) -> nn.Module:
    """
    加载 Ultralytics YOLO 模型文件，并返回底层的 PyTorch nn.Module。
    这个函数专门用于需要直接获取 torch.nn.Module 实例的场景;例如，检查或特定集成。

    Returns:
        torch.nn.Module: 底层的 PyTorch 模型实例。
    """
    try:
        # 检查模型路径是否存在
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"模型文件错误，请修改加载路径: {model_path}")

        # 自动选择设备为GPU
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 验证设备是否可用
        if device == 'cuda' and not torch.cuda.is_available():
            print("警告: CUDA不可用，将使用CPU")
            device = 'cpu'

        # 1. 先加载 YOLO 对象
        print(f"使用 YOLO 加载器加载: {model_path}")
        yolo_wrapper = YOLO(model_path)
        print("YOLO 加载器加载成功。")

        # 2. 从 YOLO 对象中提取底层的 PyTorch 模型
        #    通常，这个模型存储在 YOLO 对象的 .model 属性中
        print("正在提取底层的 torch.nn.Module...")
        pytorch_model = yolo_wrapper.model
        if not isinstance(pytorch_model, nn.Module):
             # 做个健壮性检查，以防未来 Ultralytics 内部结构改变
             raise TypeError(f"从YOLO对象提取的 '.model' 属性不是 torch.nn.Module 的实例，实际类型为 {type(pytorch_model)}")
        print(f"成功提取 PyTorch 模型，类型: {type(pytorch_model).__name__}")

        # 3. 将提取出的 PyTorch 模型移动到指定设备
        print(f"正在将 PyTorch 模型移动到 {device.upper()}...")
        pytorch_model.to(device)
        print(f"PyTorch 模型成功移动到 {device.upper()}")

        # 4. 返回这个底层的 PyTorch 模型
        return pytorch_model

    except Exception as e:
        raise RuntimeError(f"加载底层 PyTorch 模型失败: {str(e)}")
    

if __name__ == "__main__":
    model_path = "best_model.pt"
    device = "cpu"
    model = load_pytorch_module(model_path, device)
    model_yolo=load_model(model_path, device)
    print('原始的torch模型：',model)

    #print('加载yolo模型：',model_yolo)
