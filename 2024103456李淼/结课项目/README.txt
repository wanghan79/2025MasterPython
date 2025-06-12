这是一个人脸识别的项目
选用的数据集链接如下：https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset
首先，对数据集进行划分训练集和测试集操作（split.py），结果已给出（4个npy文件）
接着，自定义一个带有数据增强的 PyTorch 数据集类，用于图像分类模型训练和测试（utils.py）
然后，将已经保存好的 NumPy 格式训练数据加载并转换为 PyTorch 可用的数据加载器（DataLoader），以便用于训练和测试神经网络模型（download.py）
最后，进行训练（train_model.py）和测试模型（test_model.py）
注：用于人脸分类识别的卷积神经网络FaceCNN模型已给出
