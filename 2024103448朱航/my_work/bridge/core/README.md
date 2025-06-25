# Pepper Robot 核心服务模块

本目录包含了 Pepper 机器人桥接服务的核心实现，负责管理服务器、配置和日志等基础功能。

## 模块组成

### 服务器模块 (server.py)
- ZMQ服务器实现
- 请求处理和路由
- 代理管理
- 心跳检测
- 错误处理

### 配置模块 (config.py)
- 配置文件加载和保存
- 配置验证
- 默认配置管理
- 配置热重载

### 日志模块 (logging_util.py)
- 日志格式化
- 文件日志
- 控制台日志
- 日志级别管理
- 编码处理

## 服务器实现

### 1. 初始化流程
```python
def __init__(self, config_path=None, debug=False):
    # 加载配置
    self.config = load_config(config_path)
    
    # 初始化日志
    self.logger = LoggingUtil.configure_logger(
        name="PepperBridge",
        debug=debug
    )
    
    # 初始化ZMQ
    self.context = zmq.Context()
    self.socket = self.context.socket(zmq.REP)
```

### 2. 请求处理
```python
def _handle_request(self, message):
    # 解析请求
    request = json.loads(message)
    
    # 获取服务和方法
    service = request.get('service')
    method = request.get('method')
    
    # 调用对应方法
    result = self._call_method(service, method, args)
    
    # 返回结果
    return json.dumps({
        "status": "ok",
        "result": result
    })
```

### 3. 心跳检测
```python
def _start_heartbeat(self):
    while self.running:
        try:
            # 检查各代理状态
            for agent in self.agents.values():
                agent.check_status()
            time.sleep(1)
        except Exception as e:
            self.logger.error("心跳检测错误: %s", e)
```

## 配置管理

### 1. 配置文件格式
```yaml
robot:
  ip: 192.168.1.106
  naoqi_port: 9559
  zmq_port: 5555

services:
  motion: true
  tts: true
  video: true
  behavior: true
  sensor: true
```

### 2. 配置加载
```python
def load_config(path=None):
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    return DEFAULT_CONFIG
```

### 3. 配置验证
```python
def validate_config(config):
    required_fields = ['robot', 'services']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"缺少必要的配置项: {field}")
```

## 日志处理

### 1. 日志配置
```python
def configure_logger(name, debug=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # 添加控制台处理器
    ch = SafeStreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # 添加文件处理器
    fh = RotatingFileHandler('logs/pepper_bridge.log')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
```

### 2. 日志格式
```python
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## 开发指南

### 1. 添加新服务
1. 在配置中注册服务
2. 创建服务处理方法
3. 添加到路由表
4. 实现错误处理

### 2. 修改配置
1. 更新配置模式
2. 添加验证规则
3. 更新默认值
4. 处理向后兼容

### 3. 扩展日志
1. 添加新的日志处理器
2. 自定义日志格式
3. 实现日志过滤
4. 添加日志轮转

## 调试技巧

1. 启用调试模式：
```python
server = PepperBridgeServer(debug=True)
```

2. 查看详细日志：
```bash
tail -f logs/pepper_bridge.log
```

3. 测试ZMQ连接：
```python
import zmq
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")
```

## 注意事项

1. 线程安全
   - 使用线程锁保护共享资源
   - 避免死锁
   - 正确处理线程退出

2. 错误处理
   - 捕获所有可能的异常
   - 提供有意义的错误信息
   - 实现优雅的失败处理

3. 性能优化
   - 使用连接池
   - 实现请求缓存
   - 优化日志写入

4. 安全性
   - 验证请求来源
   - 限制访问权限
   - 保护敏感数据 