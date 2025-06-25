#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import zmq
import json
import time
import logging
import signal
import msvcrt
from bridge.core.logging_util import LoggingUtil
import codecs
import traceback
import Queue # 导入 Queue 模块
import threading # Import the whole module

# 设置默认编码为UTF-8
if sys.version_info[0] < 3:
    import imp
    imp.reload(sys)
    sys.setdefaultencoding('utf-8')
else:
    # Python 3 设置标准输出和标准错误的编码
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    if sys.stderr.encoding != 'utf-8':
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 直接导入NAOqi模块
try:
    import naoqi
    from naoqi import ALProxy
    NAOQI_AVAILABLE = True
    print("成功导入NAOqi模块")
except ImportError:
    NAOQI_AVAILABLE = False
    print("警告: 无法导入NAOqi模块，将使用代理模式")

from bridge.core.config import load_config, save_config
from bridge.agents.motion import MotionAgent
from bridge.agents.speech import SpeechAgent
from bridge.agents.vision import VisionAgent
from bridge.agents.behavior import BehaviorAgent
from bridge.agents.sensor import SensorAgent
from bridge.agents.system import SystemAgent
from bridge.agents.tts import TTSAgent
from bridge.agents.memory import MemoryAgent
from bridge.agents.tablet import TabletAgent
from bridge.agents.sound import SoundAgent
# 配置日志处理器，确保正确处理中文
class EncodingFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, encoding='utf-8'):
        logging.Formatter.__init__(self, fmt, datefmt)
        self.encoding = encoding
        
    def format(self, record):
        result = logging.Formatter.format(self, record)
        if isinstance(result, unicode) or (sys.version_info[0] >= 3 and isinstance(result, str)):
            try:
                return result.encode(self.encoding, 'replace')
            except:
                return result.encode('ascii', 'replace')
        return result

# 工作线程数量
NUM_WORKER_THREADS = 5 # 可以根据需要调整

# 核心桥接服务器类
# 职责：
# 1. 管理NAOqi服务代理（Motion/TTS/Vision等）
# 2. 处理ZMQ网络通信
# 3. 实现请求路由和负载均衡
# 4. 维护机器人健康状态监测
class PepperBridgeServer:
    """Pepper机器人桥接服务器

    功能特性：
    - 多协议代理管理（ALMotion/ALTextToSpeech等）
    - 基于ZMQ的异步通信机制
    - 自动心跳检测和故障恢复
    - 动态服务注册机制
    - 支持调试模式和配置热加载
    """

    def __init__(self, config_path=None, debug=False):
        """初始化桥接服务器

        参数说明：
        config_path -- 配置文件路径（默认使用内置配置）
        debug       -- 调试模式开关（True/False）

        初始化流程：
        1. 加载日志配置
        2. 读取/创建运行时配置
        3. 初始化服务代理
        4. 建立ZMQ通信上下文
        5. 构建方法路由表
        """
        # 初始化日志系统
        self.logger = LoggingUtil.configure_logger(
            name="PepperBridge",
            debug=debug
        )
        self.debug = debug
        
        # 加载配置
        if not config_path:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        self.config = load_config(config_path)
        self.logger.info("加载配置: %s", self.config)
        
        # 初始化ZMQ
        self.context = zmq.Context()
        # 使用ROUTER代替REP以支持异步处理和多客户端
        self.socket = self.context.socket(zmq.ROUTER) 
        self.socket.setsockopt(zmq.LINGER, 0)

        # 初始化音频流 ZMQ PUB Socket
        self.audio_stream_socket = self.context.socket(zmq.PUB)
        self.audio_stream_socket.setsockopt(zmq.LINGER, 0)
        
        # 运行标志
        self.running = False
        
        # 添加用于保护ZMQ发送操作的锁
        self.send_lock = threading.Lock()
        self.audio_send_lock = threading.Lock() # 音频流专用锁
        
        # 创建请求队列
        self.request_queue = Queue.Queue()
        
        # 工作线程列表
        self.worker_threads = []
        
        # 添加状态显示标志
        self.show_status = False
        
        # 添加信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # 添加按键检测线程
        self.key_thread = None
        
        # 初始化代理
        self._init_agents()
        
        # 动态构建方法映射表
        self._method_map = {}
        for service_name, agent in self.agents.iteritems():
            if hasattr(agent, 'register_methods'):
                try:
                    methods = agent.register_methods()
                    for category, func_map in methods.iteritems():
                        if category not in self._method_map:
                            self._method_map[category] = {}
                        self._method_map[category].update(func_map)
                    self.logger.info("成功注册 %s 代理的方法", service_name)
                except Exception as e:
                    self.logger.error("注册 %s 代理方法失败: %s", service_name, e)
                    self.logger.error("异常堆栈: %s", traceback.format_exc())
        


    def _init_agents(self):
        """初始化NAOqi服务代理

        代理初始化策略：
        - 按config.yaml配置加载基础服务
        - 为每个服务创建独立代理实例
        - 建立服务别名映射（如memory->sensor）
        - 统一设置调试模式
        - 执行连接测试和状态预检
        """
        try:
            # 统一代理初始化流程
            ip = self.config['robot']['ip']
            port = self.config['robot']['naoqi_port']
            
            self.logger.info("正在连接到机器人 %s:%s...", ip, port)
            
            # 初始化基础代理集合
            self.agents = {}
            
            # 创建代理实例
            agent_classes = {
                'motion': MotionAgent,
                'speech': SpeechAgent,
                'tts': TTSAgent,
                'video': VisionAgent,
                'sensor': SensorAgent,
                'behavior': BehaviorAgent,
                'system': SystemAgent,
                'tablet': TabletAgent,
                'memory': MemoryAgent,
                'tablet': TabletAgent,
                'sound': SoundAgent
            }
            
            # 记录初始化失败的代理
            failed_agents = []
            
            # 逐个初始化代理
            for name, agent_class in agent_classes.iteritems():
                try:
                    if name == 'sound':
                        # 特殊处理 SoundAgent，传递 ZMQ socket 和锁
                        agent = agent_class(ip, port, 
                                            audio_stream_socket=self.audio_stream_socket, 
                                            audio_send_lock=self.audio_send_lock,
                                            logger=self.logger)
                    else:
                        agent = agent_class(ip, port)
                        
                    agent.set_debug(self.debug)
                    agent.initialize()  # 显式初始化代理
                    self.agents[name] = agent
                    self.logger.info("成功初始化 %s 代理", name)
                except Exception as e:
                    self.logger.error("初始化 %s 代理失败: %s", name, e)
                    self.logger.error("异常堆栈: %s", traceback.format_exc())
                    failed_agents.append(name)
                    continue
            
            # 检查关键代理是否初始化成功
            critical_agents = ['motion', 'tts']
            missing_critical = [agent for agent in critical_agents if agent in failed_agents]
            
            if missing_critical:
                error_msg = "关键代理初始化失败: %s" % ", ".join(missing_critical)
                self.logger.error(error_msg)
                raise Exception(error_msg)
            
            if not self.agents:
                raise Exception("没有成功初始化任何代理")
            
            # 初始化机器人状态
            self._initialize_robot_state()
                
            # 测试连接
            self._test_say("桥接服务已启动")
                
        except Exception as e:
            self.logger.error("初始化代理失败: %s", e)
            self.logger.error("异常堆栈: %s", traceback.format_exc())
            raise

    def _signal_handler(self, signum, frame):
        """处理系统信号"""
        self.logger.info("收到系统信号，正在关闭服务器...")
        self.stop()
        sys.exit(0)
        
    def _check_key_press(self):
        """检测按键输入"""
        while self.running:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b's':  # 按's'键显示状态
                    self.show_status = not self.show_status
                    if self.show_status:
                        self.logger.info("已启用状态显示，按's'键关闭")
                    else:
                        self.logger.info("已关闭状态显示，按's'键开启")
                elif key == b'q':  # 按'q'键退出
                    self.logger.info("用户按下'q'键，正在关闭服务器...")
                    # 只设置running标志为False，避免在线程中调用sys.exit()
                    self.running = False
                    # 恢复机器人的自主生命状态
                    self._restore_autonomous_life()
            time.sleep(0.1)
            
    def _heartbeat_loop(self):
        """心跳检测线程（30秒间隔）"""
        try:
            self.logger.info("心跳线程已启动")
            
            while self.running:
                try:
                    # 检查所有核心代理连接状态
                    for service in ['motion', 'tts', 'speech', 'video', 'sensor']:
                        if service in self.agents:
                            try:
                                if self.show_status:
                                    self.logger.info("开始检查 %s 服务状态...", service)
                                
                                # 获取代理对象
                                agent = self.agents[service]
                                if self.show_status:
                                    self.logger.info("%s 代理对象类型: %s", service, type(agent).__name__)
                                
                                # 获取代理
                                try:
                                    if service == 'motion':
                                        if self.show_status:
                                            self.logger.info("尝试获取 motion 代理...")
                                        proxy = agent.get_proxy("ALMotion")
                                        if self.show_status:
                                            self.logger.info("成功获取 motion 代理")
                                        proxy.getSummary()
                                    elif service == 'tts':
                                        if self.show_status:
                                            self.logger.info("尝试获取 tts 代理...")
                                        proxy = agent.get_proxy("ALTextToSpeech")
                                        if self.show_status:
                                            self.logger.info("成功获取 tts 代理")
                                        proxy.getAvailableLanguages()
                                    elif service == 'speech':
                                        if self.show_status:
                                            self.logger.info("尝试获取 speech 代理...")
                                        proxy = agent.get_proxy("ALSpeechRecognition")
                                        if self.show_status:
                                            self.logger.info("成功获取 speech 代理")
                                        proxy.getAvailableLanguages()
                                    elif service == 'video':
                                        if self.show_status:
                                            self.logger.info("尝试获取 video 代理...")
                                        proxy = agent.get_proxy("ALVideoDevice")
                                        if self.show_status:
                                            self.logger.info("成功获取 video 代理")
                                        proxy.getCameraIndexes()
                                    elif service == 'sensor':
                                        if self.show_status:
                                            self.logger.info("尝试获取 sensor 代理...")
                                        proxy = agent.get_proxy("ALMemory")
                                        if self.show_status:
                                            self.logger.info("成功获取 sensor 代理")
                                        proxy.getDataList("Device/SubDeviceList")
                                    elif service == 'memory':
                                        if self.show_status:
                                            self.logger.info("尝试获取 memory 代理...")
                                        proxy = agent.get_proxy("ALMemory")
                                        if self.show_status:
                                            self.logger.info("成功获取 memory 代理")
                                        proxy.getDataList("Device/SubDeviceList")   
                                    elif service == 'tablet':
                                        if self.show_status:
                                            self.logger.info("尝试获取 tablet 代理...")
                                        proxy = agent.get_proxy("ALTabletService")
                                        if self.show_status:
                                            self.logger.info("成功获取 tablet 代理")
                                    elif service == 'sound':
                                        if self.show_status:
                                            self.logger.info("尝试获取 sound 代理...")
                                        proxy = agent.get_proxy("ALAudioPlayer")
                                        if self.show_status:
                                            self.logger.info("成功获取 sound 代理")

                                    if self.show_status:
                                        self.logger.info("%s 服务状态检查成功", service)
                                except AttributeError as e:
                                    self.logger.error("%s 代理对象缺少方法: %s", service, str(e))
                                    self.logger.error("代理对象可用方法: %s", dir(agent))
                                    raise
                                
                            except Exception as e:
                                self.logger.warning("心跳检查：%s 服务连接异常: %s", service, str(e))
                                self.logger.warning("异常堆栈: %s", traceback.format_exc())
                
                except Exception as e:
                    self.logger.error("心跳检查异常: %s", e)
                    self.logger.error("异常堆栈: %s", traceback.format_exc())
                
                # 休眠间隔
                for _ in range(30):
                    if not self.running:
                        break
                    time.sleep(1)
            
            self.logger.info("心跳线程已退出")
        
        except Exception as e:
            self.logger.error("心跳线程异常: %s", e)
            self.logger.error("异常堆栈: %s", traceback.format_exc())
            
    def _initialize_robot_state(self):
        """初始化机器人状态"""
        try:
            # 尝试禁用自主生命模式
            try:
                life_proxy = ALProxy("ALAutonomousLife", self.config["robot"]["ip"], self.config["robot"]["naoqi_port"])
                current_state = life_proxy.getState()
                
                self.logger.info("当前自主生命状态: %s", current_state)
                
                if current_state != "disabled":
                    self.logger.info("禁用机器人自主生命模式...")
                    
                    # 先停止当前活动
                    if current_state == "interactive" or current_state == "solitary":
                        try:
                            life_proxy.stopActivity()
                        except:
                            self.logger.warning("停止活动失败，继续进行")
                    
                    # 禁用自主生命
                    life_proxy.setState("disabled")
                    time.sleep(1)  # 等待状态变化
                    
                    # 再次检查状态
                    new_state = life_proxy.getState()
                    self.logger.info("禁用后自主生命状态: %s", new_state)
            except Exception as e:
                self.logger.warning("禁用自主生命失败: %s", e)
            
            # 唤醒机器人
            motion_proxy = ALProxy("ALMotion", self.config["robot"]["ip"], self.config["robot"]["naoqi_port"])
            if not motion_proxy.robotIsWakeUp():
                self.logger.info("唤醒机器人...")
                motion_proxy.wakeUp()
                time.sleep(1)
            
            # 设置全身刚度
            motion_proxy.setStiffnesses("Body", 1.0)
                
            # 释放平板控制权
            try:
                self.logger.info("正在尝试重置平板...")
                tablet_proxy = ALProxy("ALTabletService", self.config["robot"]["ip"], self.config["robot"]["naoqi_port"])
                
                # 尝试隐藏当前网页视图并重置平板
                try:
                    tablet_proxy.hideWebview()
                    time.sleep(0.5)
                    tablet_proxy.resetTablet()
                    self.logger.info("平板重置成功")
                except Exception as e:
                    self.logger.warning("重置平板失败: %s", e)
                
                self.logger.info("平板重置操作完成")
            except Exception as e:
                self.logger.warning("平板控制失败: %s", e)
                self.logger.warning("异常堆栈: %s", traceback.format_exc())
            
            self.logger.info("机器人状态初始化成功")
            return True
        except Exception as e:
            self.logger.error("机器人状态初始化失败: %s", e)
            return False
    
    def _restore_autonomous_life(self):
        """恢复机器人的自主生命状态"""
        try:
            if 'motion' in self.agents:
                motion_agent = self.agents['motion']
                if hasattr(motion_agent, 'restore_autonomous_life'):
                    motion_agent.restore_autonomous_life()
                    self.logger.info("已恢复机器人自主生命状态")
                else:
                    self.logger.warning("Motion代理没有restore_autonomous_life方法")
            else:
                self.logger.warning("无法恢复自主生命状态：未找到motion代理")
        except Exception as e:
            self.logger.warning("恢复自主生命状态失败: %s", e)
            self.logger.warning("异常堆栈: %s", traceback.format_exc())
    
    def _test_say(self, text):
        """测试语音服务

        Args:
            text (str): 要说的文本
        """
        try:
            if 'tts' in self.agents:
                tts_agent = self.agents['tts']
                if hasattr(tts_agent, 'say'):
                    tts_agent.say(text)
                    self.logger.info("语音测试成功: %s", text)
                else:
                    self.logger.warning("TTS代理没有say方法")
            else:
                self.logger.warning("无法进行语音测试：未找到tts代理")
        except Exception as e:
            self.logger.warning("语音测试失败: %s", e)
            self.logger.warning("异常堆栈: %s", traceback.format_exc())
    
    def _worker_loop(self):
        """工作线程的主循环，从队列获取并处理请求"""
        thread_name = threading.current_thread().name
        self.logger.info("工作线程 %s 已启动", thread_name)
        while self.running:
            task = None # 确保 task 在循环开始时未定义
            try:
                # 从队列中获取任务，设置一个超时以允许检查 self.running
                task = self.request_queue.get(block=True, timeout=1) # 等待1秒

                if task is None: # 收到哨兵值，退出循环
                    self.logger.info("工作线程 %s 收到退出信号", thread_name)
                    break

                client_identity, message = task

                # --- 开始处理请求 ---
                try:
                    start_time = time.time() # 记录开始时间
                    if self.debug:
                        self.logger.debug("工作线程 %s 开始处理来自 %s 的请求",
                                         thread_name, client_identity)

                    # 调用处理逻辑获取响应字符串
                    response_str = self._handle_request(message)
                    
                    processing_time = time.time() - start_time
                    if self.debug or processing_time > 2.0: # 记录慢请求
                        self.logger.info("工作线程 %s 处理请求耗时: %.3fs", thread_name, processing_time)

                    # 使用锁保护 ZMQ 发送操作，确保线程安全
                    send_start_time = time.time()
                    with self.send_lock:
                        self.socket.send_multipart([
                            client_identity,
                            b'',
                            response_str.encode('utf-8')
                        ])
                    send_time = time.time() - send_start_time
                    if self.debug or send_time > 0.1: # 记录慢发送
                         self.logger.info("工作线程 %s 发送响应耗时: %.3fs", thread_name, send_time)

                except zmq.ZMQError as e:
                     self.logger.error("工作线程 %s 发送响应给 %s 时 ZMQ 错误: %s",
                                      thread_name, client_identity, e)
                except Exception as e:
                    self.logger.error("工作线程 %s 处理请求时发生错误: %s",
                                     thread_name, e)
                    self.logger.error("异常堆栈: %s", traceback.format_exc())
                    # 尝试向客户端发送错误信息
                    try:
                        error_response = json.dumps({
                            "status": "error",
                            "error": "服务器内部错误: %s" % str(e)
                        })
                        with self.send_lock:
                            self.socket.send_multipart([
                                client_identity,
                                b'',
                                error_response.encode('utf-8')
                            ])
                    except Exception as send_error:
                         self.logger.error("工作线程 %s 尝试发送错误响应给 %s 时失败: %s",
                                          thread_name, client_identity, send_error)
                finally:
                    # 标记任务完成
                    if task is not None: # 仅在成功获取任务后调用
                         self.request_queue.task_done()
                # --- 结束处理请求 ---

            except Queue.Empty:
                # 队列在超时时间内为空，这很正常，继续检查 self.running
                continue
            except Exception as e:
                 # 捕获 get() 或其他可能的循环错误
                 self.logger.error("工作线程 %s 循环出错: %s", thread_name, e)
                 time.sleep(1) # 防止错误循环占用过多CPU

        self.logger.info("工作线程 %s 已退出", thread_name)

    def start(self):
        """启动服务器主循环和工作线程"""
        try:
            # 绑定ZMQ端口
            zmq_port = self.config['robot']['zmq_port']
            self.socket.bind("tcp://*:%d" % zmq_port)

            # 绑定音频流 Socket
            audio_stream_port = self.config['robot'].get('audio_stream_port', 5556) # 默认5556
            audio_stream_endpoint = "tcp://*:%d" % audio_stream_port
            self.audio_stream_socket.bind(audio_stream_endpoint)
            self.logger.info("音频流服务器已启动 (PUB模式)，监听端口: %d", audio_stream_port)
            self.logger.info("ZMQ服务器已启动 (ROUTER模式)，监听端口: %d", zmq_port)
            self.logger.info("使用 %d 个工作线程处理请求" % NUM_WORKER_THREADS)
            self.logger.info("按's'键显示/隐藏状态，按'q'键退出")
            
            # 设置运行标志
            self.running = True
            
            # 启动工作线程
            self.worker_threads = []
            for i in range(NUM_WORKER_THREADS):
                thread = threading.Thread(target=self._worker_loop, name="Worker-%d" % (i+1))
                thread.daemon = False 
                thread.start()
                self.worker_threads.append(thread)
            
            # 启动按键检测线程
            self.key_thread = threading.Thread(target=self._check_key_press)
            self.key_thread.daemon = True
            self.key_thread.start()
            
            # 主循环 - 接收请求并放入队列
            self.logger.info("服务器主循环已启动 (接收请求并入队)")
            while self.running:
                try:
                    # 使用 Poller 来非阻塞地检查是否有消息到达
                    # 这可以避免在没有消息时 recv_multipart() 长时间阻塞
                    # 并且允许更及时地检查 self.running 标志
                    poller = zmq.Poller()
                    poller.register(self.socket, zmq.POLLIN)
                    socks = dict(poller.poll(timeout=100)) # 等待 100ms

                    if self.socket in socks and socks[self.socket] == zmq.POLLIN:
                        multipart_message = self.socket.recv_multipart(zmq.NOBLOCK) # 非阻塞接收
                        if len(multipart_message) != 3:
                            self.logger.error("接收到无效的多部分消息: %s", multipart_message)
                            continue

                        client_identity = multipart_message[0]
                        try:
                            message = multipart_message[2].decode('utf-8')
                        except UnicodeDecodeError:
                             self.logger.error("无法解码来自 %s 的消息为 UTF-8", client_identity)
                             continue 

                        # 将任务放入队列
                        try:
                             self.request_queue.put((client_identity, message), block=False) # 非阻塞放入
                             if self.debug and self.request_queue.qsize() % 10 == 0:
                                 self.logger.debug("请求已入队 (来自 %s)，当前队列大小: %d", client_identity, self.request_queue.qsize())
                        except Queue.Full:
                             self.logger.warning("请求队列已满 (大小: %d)，暂时拒绝来自 %s 的请求", 
                                                 self.request_queue.qsize(), client_identity)
                             # 可以选择向客户端发送一个错误响应，告知服务器繁忙
                             try:
                                 error_response = json.dumps({"status": "error", "error": "服务器繁忙，请稍后重试"})
                                 with self.send_lock:
                                      self.socket.send_multipart([client_identity, b'', error_response.encode('utf-8')])
                             except Exception as send_busy_error:
                                 self.logger.error("发送服务器繁忙响应失败: %s", send_busy_error)
                             time.sleep(0.1) # 短暂暂停，避免CPU空转
                    # else: # Poller 超时，或者 socket 没有事件，继续循环检查 self.running
                    #    pass 

                except zmq.ZMQError as e:
                    if e.errno == zmq.ETERM or not self.running:
                        self.logger.info("ZMQ 上下文已终止或服务器停止，退出主循环。")
                        break
                    else:
                        self.logger.error("ZMQ通信错误: %s", e)
                        time.sleep(0.1) 
                except Exception as e:
                    self.logger.error("主循环接收/入队时发生错误: %s", e)
                    self.logger.error("异常堆栈: %s", traceback.format_exc())
                    time.sleep(0.1) 

        except Exception as e:
            self.logger.error("服务器启动失败: %s", e)
            self.logger.error("异常堆栈: %s", traceback.format_exc())
            # 确保在启动失败时也尝试停止
            self.stop() 
            raise
        # finally: # 不再需要 finally stop()，因为 stop() 会在各种退出路径被调用
        #    self.stop()

    def stop(self):
        """安全停止服务"""
        if not self.running: 
             self.logger.info("服务器已在停止过程中或已停止")
             return
        try:
            self.logger.info("正在启动服务器停止流程...")

            # 1. 设置运行标志为False，阻止新请求入队和工作线程获取新任务
            self.running = False

            # 2. 停止按键检测线程 (如果存在且活动)
            #    它可能会调用 self.stop()，需要处理重入
            if self.key_thread and self.key_thread.is_alive():
                 self.logger.info("按键检测线程仍在运行，它将完成停止流程")
                 # 通常按键线程检测到 self.running 为 False 后会退出
                 # 可以 join 等待一下，但要小心死锁
                 # self.key_thread.join(timeout=1.0) 
                 # return # 让按键线程完成停止

            # 3. 唤醒并停止工作线程
            self.logger.info("正在通知 %d 个工作线程退出..." % len(self.worker_threads))
            # 等待队列中的现有任务完成 (可选，设置超时)
            # self.request_queue.join() # 如果使用 task_done/join 机制
            for i in range(len(self.worker_threads)):
                try:
                    self.request_queue.put(None, block=True, timeout=1) # 阻塞放入哨兵值
                except Queue.Full:
                    self.logger.warning("放入第 %d 个停止信号到队列超时 (可能已满或线程已退出)" % (i+1))
                    # 即使放入失败，线程也会因 self.running == False 而退出

            # 等待所有工作线程实际结束
            self.logger.info("正在等待工作线程结束...")
            active_workers = []
            for thread in self.worker_threads:
                 thread.join(timeout=5.0) # 等待最多5秒
                 if thread.is_alive():
                      self.logger.warning("工作线程 %s 未能在超时内退出" % thread.name)
                 else:
                      self.logger.debug("工作线程 %s 已成功退出" % thread.name)
                 active_workers = [t for t in self.worker_threads if t.is_alive()]
            if active_workers:
                 self.logger.warning("%d 个工作线程未能正常退出" % len(active_workers))
            else:
                 self.logger.info("所有工作线程已成功退出")
            self.worker_threads = [] # 清空列表

            # 4. 恢复机器人的自主生命状态 (如果需要)
            self._restore_autonomous_life()

            # 5. 关闭音频流 Socket
            try:
                if hasattr(self, 'audio_stream_socket') and self.audio_stream_socket and not self.audio_stream_socket.closed:
                    self.logger.info("正在关闭音频流 ZMQ Socket...")
                    self.audio_stream_socket.close()
                    self.audio_stream_socket = None # 明确设为 None
            except Exception as e:
                self.logger.error("关闭音频流 Socket 时出错: %s", e)

            # 6. 关闭音频流 ZMQ Socket (如果存在且未关闭)
            if hasattr(self, 'audio_stream_socket') and self.audio_stream_socket and not self.audio_stream_socket.closed:
                self.logger.info("正在关闭音频流 ZMQ Socket...")
                try:
                    self.audio_stream_socket.close()
                    self.logger.info("音频流 ZMQ Socket 已关闭")
                except Exception as e:
                    self.logger.error("关闭音频流 ZMQ Socket 时出错: %s", e)

            # 7. 关闭主ZMQ Socket
            self.logger.info("正在关闭 ZMQ 套接字和上下文...")
            if hasattr(self, 'socket') and self.socket and not self.socket.closed:
                self.logger.debug("关闭 ZMQ socket")
                self.socket.close()
                self.socket = None # 明确设为 None
            else:
                self.logger.debug("ZMQ socket 已关闭或不存在")
               
            if hasattr(self, 'context') and self.context and not self.context.closed:
                 self.logger.debug("终止 ZMQ context")
                 self.context.term()
                 self.context = None # 明确设为 None
            else:
                 self.logger.debug("ZMQ context 已终止或不存在")

            self.logger.info("服务器已完全停止")
            
        except Exception as e:
            self.logger.error("停止服务器时发生严重错误: %s" % e)
            self.logger.error("异常堆栈: %s" % traceback.format_exc())
        finally:
             # 确保即使在 stop 内部出错，也将 running 设为 False
             self.running = False 

    def _handle_request(self, message):
        """请求处理逻辑（现在只负责处理，不负责发送）

        处理阶段：
        1. 请求解析：JSON解码和格式校验
        2. 路由查找：服务/方法双重验证
        3. 方法执行：异常捕获和结果包装

        返回：
        标准化JSON响应（包含status/result/error）
        """
        try:
            # 记录原始消息
            if self.debug:
                self.logger.debug("收到原始消息: %s", message)
            
            # 解析请求
            request = json.loads(message)
            
            # 记录请求
            if self.debug:
                self.logger.debug("解析后的请求: %s", request)
            
            # 提取请求参数
            service = request.get('service', '')
            method = request.get('method', '')
            args = request.get('args', [])
            
            # 记录请求参数
            if self.debug:
                self.logger.debug("请求参数: service=%s, method=%s, args=%s", service, method, args)
            
            # 检查服务是否存在
            if service not in self._method_map:
                error_msg = "服务不存在: %s" % service
                self.logger.error(error_msg)
                return json.dumps({
                    "status": "error",
                    "error": error_msg
                })
            
            # 检查方法是否存在
            if method not in self._method_map[service]:
                error_msg = "方法不存在: %s.%s" % (service, method)
                self.logger.error(error_msg)
                return json.dumps({
                    "status": "error",
                    "error": error_msg
                })
            
            # 获取方法
            func = self._method_map[service][method]
            
            # 调用方法 - 根据 args 类型选择调用方式
            try:
                if self.debug:
                    self.logger.debug("开始调用方法: %s.%s with args type: %s", service, method, type(args).__name__)
                    
                if isinstance(args, dict):
                    # 如果 args 是字典，使用关键字参数解包
                    self.logger.debug("Calling %s.%s with **kwargs: %r", service, method, args)
                    result = func(**args)
                elif isinstance(args, (list, tuple)):
                    # 如果 args 是列表或元组，使用位置参数解包
                    self.logger.debug("Calling %s.%s with *args: %r", service, method, args)
                    result = func(*args)
                elif args is None:
                     # 如果 args 是 None，尝试无参数调用
                     self.logger.debug("Calling %s.%s with no arguments (args was None)", service, method)
                     result = func()
                else:
                    # 其他不支持的 args 类型
                    raise TypeError("不支持的参数类型 (%s) for method %s.%s" % (type(args).__name__, service, method))
                
                # 构造成功响应
                response = {
                    "status": "ok",
                    "result": result
                }
                if self.debug:
                    self.logger.debug("方法调用成功，结果: %s", result)
            except Exception as e:
                error_msg = "执行方法失败: %s" % str(e)
                self.logger.error(error_msg)
                self.logger.error("异常堆栈: %s", traceback.format_exc())
                response = {
                    "status": "error",
                    "error": error_msg
                }
            
            # 不再在这里记录响应或发送，只返回JSON字符串
            return json.dumps(response)
            
        except json.JSONDecodeError as e:
            error_msg = "无效的JSON格式: %s" % str(e)
            self.logger.error(error_msg)
            # 返回错误信息的JSON字符串
            return json.dumps({ 
                "status": "error",
                "error": error_msg
            })
        except Exception as e:
            error_msg = "处理请求时发生错误: %s" % str(e)
            self.logger.error(error_msg)
            self.logger.error("异常堆栈: %s", traceback.format_exc())
            # 返回错误信息的JSON字符串
            return json.dumps({ 
                "status": "error",
                "error": error_msg
            })

def main():
    """主函数"""
    global server  # 使server成为全局变量
    
    parser = argparse.ArgumentParser(description="Pepper机器人桥接服务器")
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--ip", help="机器人IP地址")
    parser.add_argument("--port", type=int, help="机器人NAOqi端口")
    parser.add_argument("--zmq-port", type=int, help="ZMQ服务端口")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    args = parser.parse_args()
    
    # 加载配置
    config_path = args.config
    
    # 创建服务器
    server = PepperBridgeServer(config_path, args.debug)
    
    # 更新配置
    if args.ip:
        server.config['robot']['ip'] = args.ip
    if args.port:
        server.config['robot']['naoqi_port'] = args.port
    if args.zmq_port:
        server.config['robot']['zmq_port'] = args.zmq_port
    
    try:
        # 启动服务器
        server.start()
    except KeyboardInterrupt:
        print("\n正在停止服务器...")
    finally:
        server.stop()

if __name__ == "__main__":
    main()