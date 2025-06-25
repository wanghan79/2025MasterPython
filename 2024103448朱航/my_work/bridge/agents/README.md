# Pepper Robot 代理模块

本目录包含了 Pepper 机器人的各个功能代理模块，每个代理负责特定的功能域。

## 代理模块列表

### Motion Agent (运动控制代理)
- 文件：`motion.py`
- 功能：
  - 基础运动控制（前进、后退、转向）
  - 关节控制
  - 姿势管理
  - 运动状态监控

### Speech Agent (语音交互代理)
- 文件：`speech.py`
- 功能：
  - 文本转语音（TTS）
  - 语言设置
  - 音量控制
  - 语音参数调整

### Vision Agent (视觉功能代理)
- 文件：`vision.py`
- 功能：
  - 摄像头控制
  - 图像获取
  - 人脸检测
  - 视频流处理

### Behavior Agent (行为管理代理)
- 文件：`behavior.py`
- 功能：
  - 预设行为管理
  - 行为状态监控
  - 自定义行为加载
  - 行为参数配置

### Sensor Agent (传感器数据代理)
- 文件：`sensor.py`
- 功能：
  - 触摸传感器数据
  - 声音传感器数据
  - 其他传感器监控
  - 传感器事件处理

## 开发新代理

1. 创建新代理类：
```python
from bridge.agents.base import BaseAgent

class NewAgent(BaseAgent):
    def __init__(self, ip, port):
        super(NewAgent, self).__init__(ip, port)
        self._service_name = "ALNewService"  # NAOqi服务名称
```

2. 实现必要方法：
```python
def _create_service_proxy(self):
    """创建服务代理"""
    return self._create_proxy(self._service_name)

def initialize(self):
    """初始化代理"""
    try:
        self._proxy = self._create_service_proxy()
        return True
    except Exception as e:
        if self._debug:
            print("初始化失败: %s" % e)
        return False
```

3. 添加功能方法：
```python
def new_function(self, param):
    """新功能实现"""
    try:
        result = self._proxy.someFunction(param)
        return result
    except Exception as e:
        if self._debug:
            print("执行失败: %s" % e)
        return None
```

## 代理开发规范

1. 错误处理
   - 所有方法必须包含异常处理
   - 在debug模式下打印详细错误信息
   - 返回合适的错误状态或None

2. 初始化流程
   - 检查必要的NAOqi服务
   - 创建代理连接
   - 设置初始参数

3. 资源清理
   - 实现cleanup方法
   - 正确关闭连接
   - 释放资源

4. 调试支持
   - 支持debug模式
   - 提供状态检查方法
   - 添加日志记录

## 测试指南

1. 单元测试：
```python
def test_agent():
    agent = NewAgent("127.0.0.1", 9559)
    assert agent.initialize()
    assert agent.new_function(test_param) == expected_result
```

2. 集成测试：
```python
def test_integration():
    # 测试与其他代理的交互
    motion = MotionAgent(ip, port)
    new_agent = NewAgent(ip, port)
    # 测试协同功能
```

## 注意事项

1. 线程安全
   - 注意资源共享
   - 使用锁保护关键区域
   - 避免死锁

2. 性能优化
   - 减少不必要的服务调用
   - 缓存频繁使用的数据
   - 及时释放资源

3. 兼容性
   - 支持Python 2.7和3.x
   - 考虑不同NAOqi版本
   - 处理平台差异 