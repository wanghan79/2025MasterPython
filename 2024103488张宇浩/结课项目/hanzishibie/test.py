from load_model import load_model
#测试集测试代码
def test(model,data_path):
    model=load_model(model)#加载测试的yolo模型
    test_results=model.val(data=data_path, split='test',plots=True,save=True)  # 只在最终评估时使用测试集
    print(f"测试集mAP@0.5: {test_results.box.map50:.3f}")
    print(f"测试集mAP@0.5:0.95: {test_results.box.map:.3f}")

if __name__ == '__main__':
    test(r'runs\detect\train9\weights\best.pt',r'..\yolo_hanzi_dataset\dataset.yaml')