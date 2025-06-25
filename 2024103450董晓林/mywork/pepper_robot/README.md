# Pepper机器人智能交互系统

## 项目概述

这是一个基于Pepper机器人的智能交互系统，集成了计算机视觉、语音交互和机器人控制等功能。系统通过百度AI平台提供物体检测和图像分类能力，实现了机器人的智能感知和交互功能。

## 系统架构

### 核心组件

1. **视觉感知模块** (`pepper_detection.py`)
   - 物体检测和识别
   - 距离估算
   - 坐标转换
   - 图像处理

2. **机器人控制接口** (`pepper_client/api.py`)
   - 运动控制
   - 姿态控制
   - 语音交互
   - 传感器数据获取

3. **通信模块**
   - ZMQ协议通信
   - 状态监控
   - 错误处理

## 功能特性

### 1. 视觉功能
- 实时物体检测和跟踪
- 相机标定和距离估算
- 三维空间位置计算
- 图像分类和识别

### 2. 运动控制
- 精确位置控制
- 多自由度运动规划
- 安全保护机制
- 姿态调整

### 3. 交互功能
- 语音合成和识别
- 表情显示
- 平板显示控制
- 手势控制

## 安装说明

### 环境要求
- Python 3.x
- OpenCV
- ZMQ
- NumPy
- 其他依赖包（见requirements.txt）

### 安装步骤
1. 克隆项目代码
2. 安装依赖包：
   ```bash
   pip install -r requirements.txt
   ```
3. 配置百度AI平台API密钥
4. 配置机器人连接参数

## 使用说明

### 初始化
```python
from pepper_client.api import PepperRobotClient
from pepper_detection import PepperPhotoDetector

# 初始化机器人客户端
client = PepperRobotClient(host="localhost", port=5555)

# 初始化检测器
detector = PepperPhotoDetector(
    api_key="your_api_key",
    secret_key="your_secret_key",
    host="localhost",
    port=5555
)
```

### 基本功能使用

1. **物体检测**
```python
# 拍照并检测物体
image = detector.take_photo()
detection_result = detector.detect_objects(image)
```

2. **运动控制**
```python
# 移动到指定位置
client.move_to(x=0.5, y=0.0, theta=0.0)

# 设置速度
client.set_velocity(x=0.3, y=0.0, theta=0.0)
```

3. **语音交互**
```python
# 语音合成
client.say("你好，我是Pepper机器人")

# 设置语言
client.set_language("Chinese")
```

## 配置说明

### 1. 相机标定
- 标定数据存储在`camera_calibration.json`
- 包含像素到米的转换比例
- 支持自定义标定参数

### 2. 机器人参数
- 连接参数（host, port）
- 超时设置
- 重试策略

### 3. API配置
- 百度AI平台密钥
- API调用频率限制
- 错误处理策略

## 注意事项

1. **安全考虑**
   - 确保网络连接安全
   - 保护API密钥
   - 注意机器人运动安全

2. **性能优化**
   - 合理设置超时时间
   - 控制API调用频率
   - 优化图像处理参数

3. **错误处理**
   - 检查网络连接状态
   - 监控电池电量
   - 处理异常情况

## 常见问题

1. **连接问题**
   - 检查网络连接
   - 验证端口配置
   - 确认机器人状态

2. **检测问题**
   - 检查相机标定
   - 验证API密钥
   - 调整检测参数

3. **运动问题**
   - 检查电池状态
   - 确认运动范围
   - 验证安全设置

## 开发指南

### 代码结构
```
pepper_robot/
├── pepper_detection.py    # 视觉检测模块
├── pepper_client/         # 机器人控制接口
│   ├── api.py            # 核心API实现
│   └── utils.py          # 工具函数
├── camera_calibration.json # 相机标定数据
└── requirements.txt      # 依赖包列表
```

### 扩展开发
1. 添加新的检测算法
2. 实现自定义行为
3. 优化性能参数
4. 增加新的交互方式

## 维护说明

1. **日常维护**
   - 定期检查日志
   - 更新依赖包
   - 备份配置文件

2. **故障排除**
   - 检查错误日志
   - 验证配置参数
   - 测试基本功能

3. **性能监控**
   - 监控系统资源
   - 跟踪API使用
   - 优化响应时间

## 版本历史

- v1.0.0: 初始版本
  - 基本功能实现
  - 核心API支持
  - 基础文档

## 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 发起Pull Request

## 许可证

本项目采用MIT许可证

## 联系方式

- 项目维护者：[维护者信息]
- 问题反馈：[问题跟踪地址]
- 技术支持：[支持邮箱] 