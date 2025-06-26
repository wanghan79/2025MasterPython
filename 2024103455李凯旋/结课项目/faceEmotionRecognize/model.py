# model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

def create_model():
    # 添加L2正则化防止过拟合
    l2_reg = regularizers.l2(0.001)
    
    model = Sequential([
        # 第一组卷积层
        Conv2D(32, (3, 3), padding='same', activation='elu', input_shape=(48, 48, 1), kernel_regularizer=l2_reg),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='elu', kernel_regularizer=l2_reg),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),  # 添加Dropout层
        
        # 第二组卷积层 - 减少滤波器数量
        Conv2D(64, (3, 3), padding='same', activation='elu', kernel_regularizer=l2_reg),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='elu', kernel_regularizer=l2_reg),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),  # 添加Dropout层
        
        # 第三组卷积层 - 减少滤波器数量
        Conv2D(128, (3, 3), padding='same', activation='elu', kernel_regularizer=l2_reg),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='elu', kernel_regularizer=l2_reg),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),  # 添加Dropout层
        
        # 全连接层 - 减少神经元数量
        Flatten(),
        Dense(64, activation='elu', kernel_regularizer=l2_reg),
        BatchNormalization(),
        Dropout(0.5),  # 更高的Dropout率
        
        # 输出层
        Dense(7, activation='softmax')
    ])
    
    # 使用更低的学习率
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model