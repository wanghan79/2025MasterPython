# Bridge服务

这是Pepper机器人桥接服务的核心目录，包含服务器端代码和API接口。

## 目录结构

```
bridge/
├── agents/            # 机器人代理模块
│   ├── motion.py      # 运动控制代理
│   ├── speech.py      # 语音控制代理
│   ├── vision.py      # 视觉控制代理
│   ├── behavior.py    # 行为控制代理
│   └── sensor.py      # 传感器控制代理
├── core/              # 核心模块
│   ├── server.py      # 桥接服务器
│   └── config.py      # 配置文件
├── api.py             # Python 3.8 API接口
└── test_robot.py      # 测试脚本
```

## 主要组件

### 1. 桥接服务器 (core/server.py)

服务器端核心组件，负责：
- 初始化NAOqi代理
- 处理客户端请求
- 提供RPC服务
- 管理机器人状态
- 心跳检测

### 2. API接口 (api.py)

客户端API接口，提供：
- 运动控制接口
- 语音控制接口
- 行为控制接口
- 传感器控制接口
- 视觉控制接口

### 3. 代理模块 (agents/)

各个功能模块的代理实现：
- `motion.py`: 运动控制
- `speech.py`: 语音控制
- `vision.py`: 视觉控制
- `behavior.py`: 行为控制
- `sensor.py`: 传感器控制

### 4. 测试脚本 (test_robot.py)

用于测试所有功能的脚本，包括：
- 连接测试
- 功能测试
- 错误处理测试

## 使用方法

1. 启动服务器：
```powershell
python run_server.py --ip 192.168.1.119 --debug 
```

2. 使用API：
```python
from pepper_client.api import PepperRobotClient

# 创建客户端（连接服务器）
robot = PepperRobotClient(host="192.168.1.119", port=5555)

# 检查连接
if not robot.check_connection():
    print("无法连接到机器人服务器")
    exit(1)

# 使用API控制机器人
robot.say("你好，世界!")
robot.move_to(0.5, 0, 0)  # 前进0.5米
robot.rest()

# 关闭连接
robot.close()
```

或者使用with语句自动管理资源：
```python
from pepper_client.api import PepperRobotClient

with PepperRobotClient(host="192.168.1.119") as robot:
    if robot.check_connection():
        robot.say("你好，我是Pepper!")
```

## 开发说明

1. 添加新功能：
   - 在对应的代理模块中实现功能
   - 在API接口中添加对应的方法
   - 在服务器端的方法映射表中添加映射

2. 调试：
   - 使用`--debug`参数启动服务器
   - 查看服务器日志
   - 使用测试脚本验证功能

## 注意事项

1. 服务器端使用Python 2.7
2. 客户端使用Python 3.8+
3. 确保NAOqi SDK正确安装
4. 注意网络连接和防火墙设置 

# Pepper Robot API 使用指南

本文档介绍了如何使用 Pepper Robot 的 Python API 接口来控制机器人。

## 快速开始

### 1. 安装
```bash
pip install -r requirements.txt
```

### 2. 基本使用
```python
from pepper_client.api import PepperRobotClient

# 创建API实例
robot = PepperRobotClient(host="192.168.1.119", port=5555)

# 使用TTS功能
robot.say("你好，我是Pepper")

# 控制运动
robot.move(1.0, 0, 0)  # 前进1米
```

## API 功能说明

### 1. 运动控制
```python
# 移动到指定位置
robot.move_to(x=1.0, y=0.0, theta=0.0)

# 原地旋转
robot.turn(angle=90)  # 顺时针旋转90度

# 停止移动
robot.stop_move()

# 设置运动速度
robot.set_velocity(x=0.5, y=0.0, theta=0.0)
```

### 2. 语音功能
```python
# 说话
robot.say("你好")

# 设置语言
robot.set_language("Chinese")

# 设置音量
robot.set_volume(70)  # 设置音量为70%
```

### 3. 视觉功能
```python
# 拍照
image = robot.take_picture()

# 开始视频流
stream_id = robot.start_video_stream()

# 获取视频帧
frame = robot.get_video_frame(stream_id)

# 停止视频流
robot.stop_video_stream(stream_id)
```

### 4. 行为控制
```python
# 运行预设行为
robot.run_behavior("animations/Stand/Gestures/Hey_1")

# 停止行为
robot.stop_behavior("animations/Stand/Gestures/Hey_1")

# 获取已安装的行为列表
behaviors = robot.get_installed_behaviors()
```

### 5. 传感器数据
```python
# 获取电池电量
battery = robot.get_battery_level()

# 获取触摸传感器状态
touch = robot.get_touch_sensor_data()

# 获取声纳数据
sonar = robot.get_sonar_data()
```

## 高级用法

### 1. 异步操作
```python
# 异步执行行为
robot.run_behavior_async("animations/Stand/Gestures/Hey_1")

# 等待行为完成
robot.wait_for_behavior_completion()
```

### 2. 事件处理
```python
# 注册触摸事件回调
def on_touch(sensor_id, state):
    print(f"传感器 {sensor_id} 状态: {state}")

robot.subscribe_to_touch_event(on_touch)
```

### 3. 错误处理
```python
try:
    robot.move_to(1.0, 0.0, 0.0)
except Exception as e:
    print(f"移动失败: {e}")
    # 处理错误
```

## 配置选项

### 1. 连接设置
```python
robot = PepperRobotClient(
    host="192.168.1.119",    # 机器人IP
    port=5555,               # ZMQ端口
    timeout=5000,            # 超时时间（毫秒）
    debug=False              # 调试模式
)
```

### 2. 调试模式
```python
# 启用调试模式
robot.set_debug(True)

# 获取详细日志
robot.get_last_error()
```

## 最佳实践

### 1. 资源管理
```python
# 使用with语句自动管理资源
with PepperRobotClient() as robot:
    robot.say("自动管理连接")
```

### 2. 错误恢复
```python
def safe_move(robot, x, y, theta, retries=3):
    for i in range(retries):
        try:
            return robot.move_to(x, y, theta)
        except Exception as e:
            if i == retries - 1:
                raise
            print(f"重试 {i+1}/{retries}")
            time.sleep(1)
```

### 3. 性能优化
```python
# 批量操作
with robot.batch():
    robot.set_volume(70)
    robot.set_language("Chinese")
    robot.say("批量操作更高效")
```

## 常见问题

### 1. 连接问题
- 检查网络连接
- 验证IP地址和端口
- 确认防火墙设置

### 2. 超时问题
- 增加超时时间
- 检查网络延迟
- 使用异步操作

### 3. 编码问题
- 确保使用UTF-8编码
- 正确处理中文字符
- 检查日志编码设置

## 开发建议

1. 错误处理
   - 捕获所有可能的异常
   - 提供有意义的错误信息
   - 实现重试机制

2. 性能优化
   - 使用批量操作
   - 复用连接
   - 避免频繁请求

3. 调试技巧
   - 启用调试模式
   - 查看详细日志
   - 使用测试工具

## 示例代码

### 1. 简单对话
```python
def simple_conversation(robot):
    robot.say("你好，我是Pepper")
    time.sleep(2)
    robot.say("很高兴见到你")
    robot.run_behavior("animations/Stand/Gestures/Hey_1")
```

### 2. 巡逻程序
```python
def patrol(robot):
    points = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 90),
        (-1.0, 0.0, 180),
        (0.0, -1.0, 270)
    ]
    
    for x, y, theta in points:
        robot.move_to(x, y, theta)
        robot.say("已到达检查点")
        time.sleep(2)
```

### 3. 交互演示
```python
def interactive_demo(robot):
    robot.say("我来表演一些动作")
    
    # 运行一系列动作
    behaviors = [
        "animations/Stand/Gestures/Hey_1",
        "animations/Stand/Emotions/Positive/Happy_4",
        "animations/Stand/Waiting/Dance_1"
    ]
    
    for behavior in behaviors:
        robot.run_behavior(behavior)
        time.sleep(3)
    
    robot.say("演示结束")
``` 