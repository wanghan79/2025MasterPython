# train.py
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from preprocess import load_and_preprocess_data
from model import create_model
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 加载和预处理数据
X_train, X_test, y_train, y_test = load_and_preprocess_data('fer2013.csv')

# 创建模型
model = create_model()

# 数据增强 - 简单的增强策略
datagen = ImageDataGenerator(
    rotation_range=10,      # 随机旋转 ±10度
    zoom_range=0.1,         # 随机缩放 ±10%
    horizontal_flip=True,   # 水平翻转
    width_shift_range=0.1,  # 水平平移 ±10%
    height_shift_range=0.1  # 垂直平移 ±10%
)

# 回调函数
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# 训练模型 - 使用数据增强
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    steps_per_epoch=len(X_train) // 64,
    validation_data=(X_test, y_test),
    epochs=100,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 在测试集上评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f"测试集上的准确率: {accuracy:.2%}")

# 保存模型
os.makedirs('saved_model', exist_ok=True)
model.save('saved_model/emotion_model.keras')
print("模型已保存至 'saved_model/emotion_model.keras'")