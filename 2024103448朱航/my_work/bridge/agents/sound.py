#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import BaseAgent
from naoqi import ALProxy
import traceback
import time
import os
import socket
import shutil
import logging # Import logging
import threading
import uuid # Add missing import
import zmq
import struct
import numpy as np
import base64 # 确保导入
import json # 确保导入

# Import NAOqi specific modules
try:
    from naoqi import ALModule, ALBroker, ALProxy
    NAOQI_AVAILABLE = True
except ImportError:
    NAOQI_AVAILABLE = False
    print("Warning: naoqi library not found. Audio streaming will not work.")
    # Define dummy classes if naoqi is not available to avoid NameErrors
    class ALModule:
        pass
    class ALBroker:
        pass

# Try to import paramiko but don't fail if not available
try:
    import paramiko
    HAS_PARAMIKO = True
except ImportError:
    HAS_PARAMIKO = False
    print("Warning: paramiko library not found. SCP transfer will not be available.")

# Define the NAOqi module for handling audio stream callbacks
if NAOQI_AVAILABLE:
    class AudioStreamHandler(ALModule):
        """NAOqi模块，用于接收音频缓冲区并通过ZMQ发送。经过优化的实现，减少资源争用。"""
        def __init__(self, name, zmq_socket, send_lock, client_id, logger):
            """初始化NAOqi模块进行音频流处理
            
            Args:
                name (str): 模块名称（必须是ASCII字符串，非unicode）
                zmq_socket: 用于发送音频数据的ZMQ套接字
                send_lock: 用于同步ZMQ发送操作的线程锁
                client_id (str): 客户端标识符
                logger: 日志记录器实例
            """
            # 检查名称类型 - NAOqi需要字节字符串(str)而非unicode
            if isinstance(name, unicode):
                logger.debug("将模块名称从unicode转换为str: %r", name)
                name = name.encode('ascii')
                
            logger.debug("AudioStreamHandler.__init__ - 模块名称类型: %s, 值: %r", 
                        type(name).__name__, name)
                
            # 确保client_id也是字节字符串
            if isinstance(client_id, unicode):
                logger.debug("将client_id从unicode转换为str: %r", client_id)
                client_id = client_id.encode('utf-8')
                
            # 预编码client_id为字节类型，避免后续重复编码
            self.client_id_bytes = client_id if isinstance(client_id, str) else client_id.encode('utf-8')
            
            try:
                # 初始化基类
                ALModule.__init__(self, name)
                logger.debug("ALModule.__init__ 成功")
            except Exception as e:
                logger.error("初始化ALModule失败: %s", e, exc_info=True)
                raise
                
            # 设置实例变量
            self.zmq_socket = zmq_socket
            self.send_lock = send_lock
            self.logger = logger
            self.running = True
            self.module_name = name
            self.process_count = 0  # 计数器跟踪回调调用次数
            self.last_log_time = time.time()  # 时间戳用于限制日志频率
            self.data_bytes_sent = 0  # 跟踪发送的总字节数
            
            logger.warning("====== AudioStreamHandler初始化完成 =====")
            logger.warning("模块名: %r, 客户端ID: %s", name, client_id)
            
            # # 设置交错计数器，只处理一定比例的回调（减轻系统负载） - 暂时禁用
            # self.skip_counter = 0
            # self.skip_ratio = 0  # 默认不跳过任何样本

        def processRemote(self, nbOfChannels, nbOfSamplesByChannel, timestamp, inputBuffer):
            """ALAudioDevice用音频数据调用的回调方法 - 简化测试版"""
            
            # ** 简化测试：只打印日志确认回调被调用 **
            self.process_count += 1
            if self.process_count == 1 or self.process_count % 50 == 0: # 限制日志频率
                self.logger.critical("********* processRemote 回调被调用！(第 %d 次) *********", self.process_count)
                self.logger.critical("参数: Channels=%d, Samples=%d, Timestamp=%s, BufferSize=%d", 
                                 nbOfChannels, nbOfSamplesByChannel, str(timestamp), len(inputBuffer) if inputBuffer else 0)
            
            # ** 以下代码暂时禁用 **
            # # 检查是否应该跳过此帧（减轻系统负载）
            # self.skip_counter += 1
            # if self.skip_ratio > 0 and self.skip_counter % self.skip_ratio != 0:
            #     return
            
            # # 如果不应该跳过，重置计数器
            # self.skip_counter = 0
            
            # # 记录处理状态
            # current_time = time.time()
            # elapsed = current_time - self.last_log_time
            
            # # 间隔记录日志，避免过多输出
            # if self.process_count == 1 or self.process_count % 100 == 0 or elapsed > 1.0:
            #     self.logger.warning("音频处理：已接收 %d 个包，已发送 %.1f KB", 
            #                      self.process_count, self.data_bytes_sent / 1024.0)
            #     self.last_log_time = current_time
            
            # # 检查运行状态
            # if not self.running:
            #     return

            # # 检查ZMQ套接字是否有效
            # if not self.zmq_socket or getattr(self.zmq_socket, 'closed', False):
            #     self.logger.error("无法发送音频数据：套接字无效或已关闭")
            #     return

            # try:
            #     # 检查缓冲区
            #     buffer_size = len(inputBuffer) if inputBuffer else 0
            #     if buffer_size == 0:
            #         return
                
            #     # 发送音频数据
            #     with self.send_lock:
            #         self.zmq_socket.send_multipart([
            #             self.client_id_bytes,  # 主题
            #             inputBuffer           # 数据
            #         ])
                    
            #         # 更新发送统计
            #         self.data_bytes_sent += buffer_size
                    
            # except Exception as e:
            #     self.logger.error("发送音频数据失败: %s", e)
            #     import traceback
            #     self.logger.error("错误堆栈: %s", traceback.format_exc())

        def stop(self):
            """停止处理器"""
            self.logger.info("正在停止AudioStreamHandler '%s'", self.module_name)
            self.logger.warning("====== 音频处理完成 ====== 共处理 %d 个音频包，发送 %.1f KB数据", 
                              self.process_count, self.data_bytes_sent / 1024.0)
            self.running = False
else:
    # NAOqi不可用时的空类
    class AudioStreamHandler:
        def __init__(self, *args, **kwargs):
            print("警告: NAOqi不可用，AudioStreamHandler是个空壳.")
        def processRemote(self, *args, **kwargs):
            pass
        def stop(self):
            pass

# 添加音频轮询线程类
class AudioPollingThread(threading.Thread):
    """音频数据轮询线程，主动获取音频数据并发送到ZMQ"""
    
    def __init__(self, audio_device, zmq_socket, send_lock, client_id, logger, 
                 samplerate=16000, channels_mask=1):
        threading.Thread.__init__(self)
        self.daemon = True  # 设为守护线程
        self.audio_device = audio_device
        self.zmq_socket = zmq_socket
        self.send_lock = send_lock
        self.client_id = client_id.encode('utf-8') if isinstance(client_id, unicode) else client_id
        self.logger = logger
        self.running = True
        self.buffer_count = 0
        self.samplerate = samplerate
        self.channels_mask = channels_mask
        self.channels = bin(channels_mask).count('1') if channels_mask > 0 else 1
        self.last_log_time = time.time()
        
        # 根据采样率计算每次轮询应获取的样本数
        # 假设每10ms采集一次，则16kHz采样率每次应有160个样本
        self.samples_per_poll = int(samplerate * 0.01)
        
        # 计算采样间隔（秒）
        self.poll_interval = 0.01  # 10ms，可根据实际需求调整
        
    def run(self):
        """线程主循环，定期轮询音频数据"""
        self.logger.warning("音频轮询线程已启动 - 采样率: %d Hz, 通道掩码: %d, 轮询间隔: %.1f ms", 
                          self.samplerate, self.channels_mask, self.poll_interval * 1000)
            
        # 主轮询循环
        while self.running:
            try:
                current_time = time.time()
                
                # === 重点调试区域：尝试获取音频数据 ===
                audio_buffer = None
                buffer_source = "unknown"
                get_data_success = False # 标记是否成功获取到数据
                
                # 方法列表
                try_methods = [
                    ("getFrontMicrophoneData", []),
                    ("getMicrophoneData", [0]),
                ]
                
                # 尝试每种方法直到成功
                for method_name, args in try_methods:
                    if not hasattr(self.audio_device, method_name):
                        # 如果方法不存在，记录并跳过
                        if current_time - self.last_log_time > 5.0: # 限制日志频率
                             self.logger.debug("轮询调试: 方法 %s 不存在于 audio_device", method_name)
                        continue
                        
                    try:
                        self.logger.debug("轮询调试: 尝试调用方法 %s(%s)", method_name, args if args else "")
                        if args:
                            audio_buffer = getattr(self.audio_device, method_name)(*args)
                        else:
                            audio_buffer = getattr(self.audio_device, method_name)()
                            
                        # 检查获取到的数据
                        if audio_buffer and len(audio_buffer) > 0:
                            buffer_source = method_name
                            get_data_success = True # 标记成功
                            self.logger.warning("轮询调试: 方法 %s 成功获取 %d 字节数据", method_name, len(audio_buffer))
                            break # 获取到数据就停止尝试
                        else:
                            # 方法调用成功，但返回空数据
                            self.logger.debug("轮询调试: 方法 %s 调用成功，但返回空数据或None (类型: %s)", method_name, type(audio_buffer).__name__)
                            
                    except Exception as method_e:
                        # 方法调用失败
                        self.logger.warning("轮询调试: 调用方法 %s 失败: %s", method_name, method_e)
                        # 可以在这里添加更详细的 traceback 打印 (如果需要)
                        # import traceback
                        # self.logger.debug("轮询调试: %s 失败堆栈:\n%s", method_name, traceback.format_exc())
                        pass # 继续尝试下一个方法
                # === 调试区域结束 ===
                
                # 如果所有方法都失败，生成静音
                if not get_data_success:
                    # 创建一帧静音数据 (16位PCM，值为0)
                    silence_samples = self.samples_per_poll * self.channels
                    audio_buffer = struct.pack('<' + 'h' * silence_samples, *([0] * silence_samples))
                    buffer_source = "silence"
                    # 限制日志频率
                    if current_time - self.last_log_time > 5.0:
                        self.logger.warning("轮询调试: 未能从任何方法获取音频数据，生成 %d 字节静音帧", len(audio_buffer))
                        
                
                # *** 数据格式化为 JSON + Base64 *** (这部分代码保持不变)
                try:
                    audio_base64 = base64.b64encode(audio_buffer).decode('utf-8')
                    message_data = {
                        "audio_data": audio_base64,
                        "timestamp": time.time(),
                        "source": buffer_source
                    }
                    json_message = json.dumps(message_data)
                    message_bytes = json_message.encode('utf-8')
                except Exception as format_e:
                    self.logger.error("格式化音频数据失败: %s", format_e)
                    continue
                    
                # 发送到ZMQ (这部分代码保持不变)
                self.buffer_count += 1
                with self.send_lock:
                    self.zmq_socket.send_multipart([
                        self.client_id,
                        message_bytes
                    ])
                
                # 日志记录 (这部分代码保持不变)
                if self.buffer_count % 100 == 0 or current_time - self.last_log_time > 5.0:
                    audio_len = len(audio_buffer) if audio_buffer else 0
                    log_level = logging.WARNING if get_data_success else logging.INFO # 成功获取数据时用WARNING醒目点
                    self.logger.log(log_level, "音频轮询进度 - 已发送 %d 帧，当前原始大小: %d 字节，来源: %s", 
                                       self.buffer_count, audio_len, buffer_source)
                    self.last_log_time = current_time
                
                # 休眠 (这部分代码保持不变)
                processing_time = time.time() - current_time
                sleep_duration = max(0, self.poll_interval - processing_time)
                time.sleep(sleep_duration)
                
            except Exception as e:
                self.logger.error("音频轮询出错: %s", e, exc_info=True)
                time.sleep(0.1)
        
        self.logger.warning("音频轮询线程已停止，共发送 %d 帧", self.buffer_count)
        
    def stop(self):
        """停止轮询线程"""
        self.running = False

class SoundAgent(BaseAgent):
    """Pepper robot audio agent for recording and streaming sounds"""

    def __init__(self, ip, port, audio_stream_socket=None, audio_send_lock=None, logger=None):
        """Initialize audio agent

        Args:
            ip (str): Robot IP address
            port (int): NAOqi port
            audio_stream_socket (zmq.Socket): ZMQ PUB socket for audio streaming.
            audio_send_lock (threading.Lock): Lock for protecting ZMQ send operations.
            logger (logging.Logger): Logger instance.
        """
        super(SoundAgent, self).__init__(ip, port)
        # Always use forward slashes for robot paths
        self._robot_recordings_dir = '/home/nao/recordings'
        self.audio_stream_socket = audio_stream_socket
        self.audio_send_lock = audio_send_lock
        self.streaming_clients = {} # Stores info about active streams {client_id: {'handler': handler_instance, 'module_name': module_name}}
        self.broker = None
        self.audio_device = None
        self.broker_ip = "0.0.0.0" # Listen on all interfaces for broker
        self.broker_port = 0 # Let the system choose a free port
        self._initialized = False  # 标记初始化状态

        # Use provided logger or create a new one
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger("SoundAgent")
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)

        # Update logger level based on debug flag
        if getattr(self, '_debug', False):
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        # 不在构造函数中自动初始化，而是等待server明确调用initialize
        # 这可以防止多次初始化
        # if NAOQI_AVAILABLE:
        #    self.initialize()
        # else:
        #    self.logger.warning("NAOqi library not found, audio streaming disabled.")

    def initialize(self):
        """Initialize NAOqi broker and audio device proxy for streaming."""
        if not NAOQI_AVAILABLE:
            self.logger.warning("Cannot initialize NAOqi components: library not available.")
            return False

        # 防止重复初始化
        if self._initialized and self.broker is not None and self.audio_device is not None:
            self.logger.warning("SoundAgent已经初始化，跳过重复初始化")
            return True

        try:
            # 检查是否已有broker实例
            if self.broker is not None:
                self.logger.warning("NAOqi broker已存在，跳过broker创建")
                # 仍然尝试初始化音频设备
                if self.audio_device is None:
                    try:
                        self.audio_device = ALProxy("ALAudioDevice", self.ip, self.port)
                        self.logger.info("Successfully connected to ALAudioDevice.")
                    except Exception as e:
                        self.logger.error("Failed to connect to ALAudioDevice: %s", e)
                self._initialized = True
                return True

            # Create a NAOqi broker for handling callbacks
            # Use a unique name to avoid conflicts if multiple agents run
            broker_name = "SoundAgentBroker_{0}".format(uuid.uuid4().hex[:8])
            self.logger.info("Starting NAOqi broker '{0}' on {1}:{2}".format(broker_name, self.broker_ip, self.broker_port))
            self.broker = ALBroker(broker_name, self.broker_ip, self.broker_port, self.ip, self.port)
            self.logger.info("NAOqi broker '%s' started successfully." % broker_name)

            # Get proxy to ALAudioDevice
            self.audio_device = ALProxy("ALAudioDevice", self.ip, self.port)
            self.logger.info("Successfully connected to ALAudioDevice.")
            
            self._initialized = True
            return True

        except Exception as e:
            self.logger.error("Failed to initialize NAOqi components: %s", e, exc_info=True)
            self.broker = None
            self.audio_device = None
            self._initialized = False
            return False

    def register_methods(self):
        """注册供客户端调用的方法"""
        methods = {
            "sound": {
                "set_recordings_dir": self.set_recordings_dir,
                "get_recordings_dir": self.get_recordings_dir,
                "start_recording": self.start_recording,
                "post_start_recording": self.post_start_recording,
                "stop_recording": self.stop_recording,
                "record_audio": self.record_audio,
                "transfer_file": self.transfer_file,
                "transfer_file_scp": self.transfer_file_scp,
                "record_and_transfer": self.record_and_transfer,
                "check_connection": self.check_connection,
                "get_audio_device_info": self.get_audio_device_info
            }
        }
        # Add streaming methods only if NAOqi is available
        if NAOQI_AVAILABLE and self.broker and self.audio_device:
            methods["sound"].update({
                "start_audio_stream": self.start_audio_stream,
                "stop_audio_stream": self.stop_audio_stream
            })
        else:
             self.logger.warning("Audio streaming methods not registered because NAOqi is unavailable or initialization failed.")
        return methods

    def _create_service_proxy(self):
        """Create audio service proxy"""
        # ALAudioRecorder is needed for recording
        self._proxies['ALAudioRecorder'] = ALProxy("ALAudioRecorder", self.ip, self.port)
        # ALAudioDevice is needed for streaming, but initialized separately
        # We don't add self.audio_device to self._proxies to avoid shutdown conflicts

    def _get_audio_recorder(self):
        """Get ALAudioRecorder proxy"""
        return self.get_proxy("ALAudioRecorder")

    def start_recording(self, filename, format_type='wav', samplerate=16000, channels=(1, 1, 1, 1)):
        """Start recording

        Args:
            filename (str): Path to save the recording (use forward slashes)
            format_type (str, optional): File format, default wav
            samplerate (int, optional): Sample rate, default 16000
            channels (tuple or list, optional): Channel settings, format (front, rear, left, right). Will be converted to tuple.

        Returns:
            bool: Success or failure
        """
        try:
            recorder = self._get_audio_recorder()
            
            # 严格类型转换和路径处理
            robot_path = str(filename).replace('\\', '/').strip()
            # NAOqi需要的具体格式，不要转换为小写
            format_type = str(format_type).strip()  
            # 确保格式是'wav'而不是其他变体
            if format_type.lower() == 'wav':
                format_type = 'wav'  # 使用标准小写格式
            samplerate = int(samplerate)
            channels_tuple = tuple(int(c) for c in channels)  # 确保通道值是整数
            
            # 确保路径不是空的
            if not robot_path:
                self.logger.error("Empty path provided for recording")
                return False
                
            # 确保文件扩展名与格式匹配
            file_ext = os.path.splitext(robot_path)[1].lower()
            if not file_ext:  # 如果没有扩展名
                robot_path = robot_path + '.' + format_type
                self.logger.debug("Added extension to path: {}".format(robot_path))
            elif file_ext[1:] != format_type.lower():  # 如果扩展名不匹配格式
                self.logger.warning("File extension {} doesn't match format {}".format(file_ext, format_type))
                # 可能要替换扩展名，但这里先保留原样
            
            # 检查目录是否存在
            robot_dir = os.path.dirname(robot_path)
            if robot_dir and not os.path.exists(robot_dir):
                self.logger.warning("Directory may not exist: {}".format(robot_dir))
            
            self.logger.debug("Calling startMicrophonesRecording with:")
            self.logger.debug("  path='{}' (type={})".format(robot_path, type(robot_path)))
            self.logger.debug("  format='{}' (type={})".format(format_type, type(format_type)))
            self.logger.debug("  rate={} (type={})".format(samplerate, type(samplerate)))
            self.logger.debug("  channels={} (type={})".format(channels_tuple, type(channels_tuple)))
            
            # NAOqi文档中的参数描述
            self.logger.debug("NAOqi API signature: startMicrophonesRecording(const std::string& filename, const std::string& type, const int& sampleRate, const AL::ALValue& channels)")
            
            # 尝试一下不同的调用方式
            try:
                # 方法1：使用原始参数格式
                recorder.startMicrophonesRecording(robot_path, format_type, samplerate, channels_tuple)
                self.logger.info("Recording started successfully (method 1)")
            except Exception as e1:
                self.logger.warning("Method 1 failed: {}".format(e1))
                try:
                    # 方法2：明确为字符串指定'wav'格式
                    recorder.startMicrophonesRecording(robot_path, 'wav', samplerate, channels_tuple)
                    self.logger.info("Recording started successfully (method 2)")
                except Exception as e2:
                    self.logger.warning("Method 2 failed: {}".format(e2))
                    try:
                        # 方法3：直接调用，不使用exec
                        self.logger.info("Trying method 3 - direct call...")
                        # 创建 ALValue 对象显式传递通道参数
                        proxy = self._proxies['ALAudioRecorder']
                        # 直接调用代理而不使用exec
                        proxy.startMicrophonesRecording(str(robot_path), str(format_type), int(samplerate), channels_tuple)
                        self.logger.info("Recording started successfully (method 3)")
                    except Exception as e3:
                        self.logger.error("Method 3 also failed: {}".format(e3))
                        raise e3  # 重新抛出最后的异常
            
            return True
        except Exception as e:
            self.logger.error("Start recording error: %s", e, exc_info=True)
            return False

    def post_start_recording(self, filename, format_type='wav', samplerate=16000, channels=(1, 1, 1, 1)):
        """Start recording asynchronously

        Args:
            filename (str): Path to save the recording (use forward slashes)
            format_type (str, optional): File format, default wav
            samplerate (int, optional): Sample rate, default 16000
            channels (tuple or list, optional): Channel settings, format (front, rear, left, right). Will be converted to tuple.

        Returns:
            bool: Success or failure
        """
        try:
            recorder = self._get_audio_recorder()
            
            # 严格类型转换和路径处理
            robot_path = str(filename).replace('\\', '/').strip()
            format_type = str(format_type).strip().lower()  # 确保小写和无多余空格
            samplerate = int(samplerate)
            channels_tuple = tuple(channels)
            
            # 确保路径不是空的
            if not robot_path:
                self.logger.error("异步录音提供了空路径")
                return False
                
            # 确保路径格式正确（以 / 开头）
            if not robot_path.startswith('/'):
                self.logger.warning("异步录音路径不以 / 开头，可能会有问题: {}".format(robot_path))
                # 可能需要根据实际情况添加前缀
            
            self.logger.debug("调用 post.startMicrophonesRecording:")
            self.logger.debug("  路径='{}'（类型={}）".format(robot_path, type(robot_path)))
            self.logger.debug("  格式='{}'（类型={}）".format(format_type, type(format_type)))
            self.logger.debug("  采样率={}（类型={}）".format(samplerate, type(samplerate)))
            self.logger.debug("  通道={}（类型={}）".format(channels_tuple, type(channels_tuple)))
            
            # 尝试一下不同的调用方式
            try:
                # 方法1：直接调用
                recorder.post.startMicrophonesRecording(robot_path, format_type, samplerate, channels_tuple)
                self.logger.info("异步录音启动成功（方法1）")
            except Exception as e1:
                self.logger.warning("异步录音方法1失败: {}".format(e1))
                try:
                    # 方法2：尝试强制字符串转换
                    recorder.post.startMicrophonesRecording(str(robot_path), str(format_type), int(samplerate), tuple(channels_tuple))
                    self.logger.info("异步录音启动成功（方法2）")
                except Exception as e2:
                    self.logger.warning("异步录音方法2失败: {}".format(e2))
                    # 最后一个尝试
                    self.logger.info("尝试异步录音方法3...")
                    try:
                        recorder.post.startMicrophonesRecording(robot_path, format_type, samplerate, channels_tuple)
                        self.logger.info("异步录音启动成功（方法3）")
                    except Exception as e3:
                        self.logger.error("异步录音方法3也失败: {}".format(e3))
                        raise e3  # 重新抛出最后的异常
            
            return True # 异步调用的返回值，True表示调用已发送
        except Exception as e:
            self.logger.error("异步录音启动错误: %s", e, exc_info=True)
            return False

    def stop_recording(self):
        """Stop recording

        Returns:
            bool: Success or failure
        """
        try:
            recorder = self._get_audio_recorder()
            self.logger.debug("Calling stopMicrophonesRecording...")
            recorder.stopMicrophonesRecording()
            self.logger.info("Successfully stopped recording.")
            return True
        except Exception as e:
            # Log error explicitly using the logger
            self.logger.error("Stop recording error: %s", e, exc_info=True) # Add exc_info=True
            # if self._debug:
            #     print("Stop recording error: %s" % e)
            #     print(traceback.format_exc())
            return False

    def set_recordings_dir(self, dir_path):
        """Set the directory for storing recordings on the robot

        Args:
            dir_path (str): Recording storage path (use forward slashes)
        """
        # Always store with forward slashes
        self._robot_recordings_dir = dir_path.replace('\\', '/')

    def get_recordings_dir(self):
        """Get the directory for storing recordings on the robot

        Returns:
            str: Recording storage path
        """
        return self._robot_recordings_dir

    def record_audio(self, filename, duration, format_type='wav', samplerate=16000, channels=(1, 0, 0, 0)):
        """Record audio for a specified duration

        Args:
            filename (str): Filename (without path)
            duration (float): Recording duration (seconds)
            format_type (str, optional): File format, default wav
            samplerate (int, optional): Sample rate, default 16000
            channels (tuple or list, optional): Channel settings, format (front, rear, left, right). Will be converted to tuple.

        Returns:
            bool: Success or failure
            str: Full path of the recording on the robot (using forward slashes) or empty string on failure
        """
        robot_file_path = "" # Initialize
        try:
            # 严格类型转换
            filename = str(filename).strip()
            # NAOqi需要的具体格式，不要转换为小写
            format_type = str(format_type).strip()
            # 确保格式是'wav'而不是其他变体
            if format_type.lower() == 'wav':
                format_type = 'wav'  # 使用标准小写格式
            duration = float(duration)
            samplerate = int(samplerate)
            channels_tuple = tuple(int(c) for c in channels)  # 确保通道值是整数
            
            # 检查文件名是否已包含扩展名
            name_parts = os.path.splitext(filename)
            base_name = name_parts[0]
            file_ext = name_parts[1].lower()
            
            # 确保文件名安全（不含特殊字符）
            safe_base_name = base_name.replace('\\', '_').replace(':', '_').replace('/', '_')
            
            # 添加正确的扩展名（如果需要）
            if not file_ext:
                safe_filename = safe_base_name + '.' + format_type
            elif file_ext[1:] != format_type.lower():
                # 扩展名与格式不匹配，替换为正确的扩展名
                self.logger.warning("Replacing extension {} with {} based on format type".format(file_ext, format_type))
                safe_filename = safe_base_name + '.' + format_type
            else:
                # 扩展名正确
                safe_filename = safe_base_name + file_ext
            
            # 构建绝对路径（使用正斜杠）
            robot_dir = str(self._robot_recordings_dir).replace('\\', '/')
            if not robot_dir.endswith('/'):
                robot_dir += '/'
                
            # 确保目录以 / 开头（NAOqi 可能需要绝对路径）
            if not robot_dir.startswith('/'):
                robot_dir = '/' + robot_dir
                
            robot_file_path = robot_dir + safe_filename
            
            self.logger.info("准备录音: 文件='{}'，时长={:.2f}秒，采样率={}，通道={}".format(
                robot_file_path, duration, samplerate, channels_tuple))
                
            # 添加额外的日志调试信息
            self.logger.debug("路径详情:")
            self.logger.debug("  原始文件名: {}".format(filename))
            self.logger.debug("  基本名称: {}".format(base_name))
            self.logger.debug("  原始扩展名: {}".format(file_ext))
            self.logger.debug("  安全文件名: {}".format(safe_filename))
            self.logger.debug("  机器人目录: {}".format(robot_dir))
            self.logger.debug("  完整路径: {}".format(robot_file_path))
            self.logger.debug("  格式: {}".format(format_type))
            
            # 检查参数类型
            self.logger.debug("参数类型:")
            self.logger.debug("  路径类型: {}".format(type(robot_file_path)))
            self.logger.debug("  格式类型: {}".format(type(format_type)))
            self.logger.debug("  采样率类型: {}".format(type(samplerate)))
            self.logger.debug("  通道类型: {}".format(type(channels_tuple)))

            # 开始录音
            self.logger.info("开始录音过程...")
            if not self.start_recording(robot_file_path, format_type, samplerate, channels_tuple):
                self.logger.error("启动录音失败: '{}'".format(robot_file_path))
                return False, ""

            # 等待指定的时长
            self.logger.info("录音进行中: {:.2f}秒...".format(duration))
            time.sleep(duration)
            self.logger.info("录音时间已到.")

            # 停止录音
            self.logger.info("尝试停止录音...")
            if not self.stop_recording():
                self.logger.error("停止录音失败: '{}', 但录音可能已经完成.".format(robot_file_path))
                # 考虑返回 True？或特定错误？目前返回 False
                return False, ""

            # 检查文件是否实际存在
            self.logger.info("录音过程完成: {}".format(robot_file_path))
            
            # 可以在这里添加文件存在检查，但由于是在远程机器人上，可能无法直接检查
            
            return True, robot_file_path

        except Exception as e:
            # 使用 logger 记录错误
            self.logger.error("录音过程中发生异常: %s", e, exc_info=True)
            
            # 如果在开始录音后发生错误，尝试停止录音
            try:
                # 检查录音代理是否存在，否则停止没有意义
                if 'ALAudioRecorder' in self._proxies:
                   self.logger.warning("由于录音过程中出错，尝试停止录音.")
                   self.stop_recording()
            except Exception as stop_e:
                 self.logger.error("错误处理过程中停止录音失败: {}".format(stop_e), exc_info=True)

            return False, "" # 失败时返回空路径

    def transfer_file_scp(self, robot_file, local_dir, username='nao', password='pepper2023'): # <-- Use correct password
        """Transfer file from robot to computer using SCP (requires paramiko library)

        Args:
            robot_file (str): File path on robot
            local_dir (str): Local save directory
            username (str, optional): Robot SSH username
            password (str, optional): Robot SSH password

        Returns:
            bool: Success or failure
            str: Local saved file path or error message
        """
        if not HAS_PARAMIKO:
            msg = "paramiko library not installed, cannot use SCP transfer"
            if self._debug:
                print(msg)
            return False, msg

        ssh = None # Initialize ssh variable
        sftp = None # Initialize sftp variable
        try:
            # Ensure local directory exists
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)

            filename = os.path.basename(robot_file)
            local_file = os.path.join(local_dir, filename)

            if self._debug:
                print("Transferring file via SCP: %s -> %s" % (robot_file, local_file))

            # Create SSH client
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Connect to robot using the provided password
            if self._debug:
                print("Connecting to SSH with username='{}' and password='{}'".format(username, '*'*len(password))) # Mask password in logs
            ssh.connect(self.ip, username=username, password=password, timeout=10)

            # Create SFTP client
            sftp = ssh.open_sftp()

            # Download file
            if self._debug:
                 print("Attempting to get file via SFTP: {} to {}".format(robot_file, local_file))
            sftp.get(robot_file, local_file)

            if self._debug:
                print("SCP file transfer complete: %s" % local_file)

            return True, local_file

        except paramiko.AuthenticationException as auth_e:
            error_msg = "SCP Authentication Error: %s. Check username ('%s') and password." % (auth_e, username)
            if self._debug:
                print(error_msg)
            return False, "Authentication failed" # Return generic message
        except Exception as e:
            error_msg = "SCP file transfer error: %s" % e
            if self._debug:
                print(error_msg)
                print(traceback.format_exc())
            # Return the error message for better diagnosis in the calling script
            return False, error_msg
        finally:
            # Ensure connections are closed
            if sftp:
                try:
                    sftp.close()
                except: pass
            if ssh:
                try:
                     ssh.close()
                except: pass

    def transfer_file(self, robot_file, local_dir, username='nao', password='pepper2023'): # <-- Use correct password
        """使用SCP(SFTP)方式将文件从机器人传输到计算机

        Args:
            robot_file (str): File path on robot
            local_dir (str): Local save directory
            username (str, optional): Robot username
            password (str, optional): Robot password

        Returns:
            bool: Success or failure
            str: Local saved file path or error message
        """
        if not HAS_PARAMIKO:
            error_msg = "paramiko库未安装，无法使用SCP传输。请安装paramiko: pip install paramiko"
            self.logger.error(error_msg)
            return False, error_msg
        
        # 直接使用SCP方式传输
        self.logger.info("使用SCP传输文件...")
        return self.transfer_file_scp(robot_file, local_dir, username, password)

    def record_and_transfer(self, filename, duration, local_dir,
                            format_type='wav', samplerate=16000, channels=(1, 0, 0, 0),
                            username='nao', password='pepper2023'): # <-- Add password here
        """Record audio and transfer to computer

        Args:
            filename (str): Filename (without path)
            duration (float): Recording duration (seconds)
            local_dir (str): Local save directory
            format_type (str, optional): File format, default wav
            samplerate (int, optional): Sample rate, default 16000
            channels (tuple or list, optional): Channel settings, format (front, rear, left, right). Will be converted to tuple.
            username (str, optional): Robot username for transfer
            password (str, optional): Robot password for transfer

        Returns:
            bool: Success or failure
            str: Local saved file path or error message
        """
        try:
            self.logger.info("===== 开始录制并传输流程 =====")
            self.logger.info("文件名: '{}', 目标目录: '{}'".format(filename, local_dir))
            
            # 严格类型转换
            filename = str(filename).strip()
            duration = float(duration)
            local_dir = str(local_dir).strip()
            # NAOqi需要的具体格式，不要转换为小写
            format_type = str(format_type).strip()
            # 确保格式是'wav'而不是其他变体
            if format_type.lower() == 'wav':
                format_type = 'wav'  # 使用标准小写格式
            samplerate = int(samplerate)
            channels_tuple = tuple(int(c) for c in channels)  # 确保通道值是整数
            username = str(username)
            password = str(password)
            
            # 参数详细记录
            self.logger.debug("录音参数详情:")
            self.logger.debug("  文件名: {}".format(filename))
            self.logger.debug("  时长: {}秒".format(duration))
            self.logger.debug("  本地目录: {}".format(local_dir))
            self.logger.debug("  格式: {}".format(format_type))
            self.logger.debug("  采样率: {}".format(samplerate))
            self.logger.debug("  通道: {}".format(channels_tuple))
            self.logger.debug("  用户名: {}".format(username))
            
            # 执行录音
            self.logger.info("开始执行录音...")
            recording_success, robot_file = self.record_audio(filename, duration, format_type, samplerate, channels_tuple)
            
            # 对录音结果进行详细检查
            if not recording_success:
                self.logger.error("录音失败")
                if not robot_file:
                    return False, "录音失败，没有生成文件"
                else:
                    self.logger.warning("虽然报告录音失败，但可能有生成文件: {}".format(robot_file))
                    # 继续尝试传输，以防万一文件实际存在
            elif not robot_file:
                self.logger.error("录音报告成功，但没有返回文件路径")
                return False, "录音成功但没有文件路径"
            
            # 检查录音文件路径
            if not robot_file:
                self.logger.error("没有有效的录音文件路径")
                return False, "没有有效的录音文件路径"
            
            self.logger.info("录音阶段完成。机器人文件路径: '{}'".format(robot_file))
            
            # 等待更长时间，确保文件保存完成
            wait_time = 0.2  # 增加到0.2秒
            self.logger.info("录音在机器人上完成 ('{}'). 等待 {:.1f} 秒后开始传输...".format(robot_file, wait_time))
            time.sleep(wait_time)
            
            # 检查文件扩展名
            file_ext = os.path.splitext(robot_file)[1].lower()
            if not file_ext:
                self.logger.warning("机器人文件没有扩展名: {}".format(robot_file))
                # 添加扩展名
                robot_file = robot_file + '.' + format_type
                self.logger.info("已添加扩展名: {}".format(robot_file))
            elif file_ext[1:] != format_type.lower():
                self.logger.warning("文件扩展名 {} 与格式类型 {} 不匹配".format(file_ext, format_type))
            
            # 传输文件（直接使用SCP传输方法）
            self.logger.info("开始尝试将文件 '{}' 传输到 '{}'".format(robot_file, local_dir))
            
            # 使用SCP传输方法
            self.logger.info("使用SCP传输方法...")
            transfer_success, transfer_detail = self.transfer_file(robot_file, local_dir, username, password)
            
            # 检查传输结果
            if transfer_success:
                # 确认文件确实存在于本地
                if isinstance(transfer_detail, str) and os.path.exists(transfer_detail):
                    self.logger.info("文件传输成功。本地路径: {}".format(transfer_detail))
                    file_size = os.path.getsize(transfer_detail)
                    self.logger.info("文件大小: {} 字节".format(file_size))
                    
                    # 检查文件类型
                    try:
                        if file_size > 0:
                            with open(transfer_detail, 'rb') as f:
                                header = f.read(10)  # 读取文件头部
                                self.logger.debug("文件头部字节: {}".format(repr(header)))
                                if header.startswith(b'{'):
                                    self.logger.warning("警告：文件似乎是JSON格式而不是WAV！")
                                    # 尝试读取并解析JSON内容
                                    try:
                                        with open(transfer_detail, 'r') as json_f:
                                            import json
                                            content = json.load(json_f)
                                            self.logger.warning("JSON内容: {}".format(content))
                                    except Exception as json_e:
                                        self.logger.warning("尝试解析JSON内容时出错: {}".format(json_e))
                                elif header.startswith(b'RIFF'):
                                    self.logger.info("文件是有效的WAV格式")
                                else:
                                    self.logger.warning("文件不是标准WAV格式，头部: {}".format(repr(header)))
                        else:
                            self.logger.warning("文件大小为零！")
                    except Exception as check_e:
                        self.logger.warning("检查文件类型时出错: {}".format(check_e))
                    
                    return True, transfer_detail
                else:
                    self.logger.warning("传输报告成功，但本地找不到文件: {}".format(transfer_detail))
                    return False, "传输报告成功，但找不到文件"
            else:
                self.logger.error("文件传输失败。详情: {}".format(transfer_detail))
                # 返回传输错误详情
                return False, "传输失败: {}".format(transfer_detail)
        
        except Exception as e:
            self.logger.error("录音和传输过程中发生异常: %s", e, exc_info=True)
            return False, "处理异常: {}".format(str(e))

    def check_connection(self):
        """Check connection status with robot

        Returns:
            bool: Connection status
        """
        try:
            # First try to establish connection to NAOqi port
            socket.create_connection((self.ip, self.port), timeout=2)
            return True
        except Exception as e:
            if self._debug:
                print("NAOqi connection check failed: %s" % e)
            return False

    def get_audio_device_info(self):
        """获取音频设备信息
        
        Returns:
            dict: 包含音频设备状态信息的字典
        """
        try:
            result = {
                "broker_active": self.broker is not None,
                "audio_device_ready": self.audio_device is not None,
                "streams_count": len(self.streaming_clients),
                "active_streams": list(self.streaming_clients.keys()),
                "naoqi_available": NAOQI_AVAILABLE
            }
            
            # 添加更多设备信息（如果音频设备可用）
            if self.audio_device:
                try:
                    result["sample_rate"] = self.audio_device.getOutputRate()
                except:
                    # 如果获取采样率失败，不要中断整个方法
                    result["sample_rate"] = "未知"
                    
            self.logger.debug("获取音频设备信息: %s", result)
            return result
        except Exception as e:
            self.logger.error("获取音频设备信息失败: %s", e, exc_info=True)
            return {"error": str(e), "broker_active": self.broker is not None}

    def start_audio_stream(self, client_id, channels_mask=1, frequency=16000, buffer_size=1024, sample_rate=None, **kwargs):
        """启动音频流

        Args:
            client_id: 请求音频流的客户端ID
            channels_mask: 通道掩码，默认为1（仅前麦克风）
                          位掩码值：1=前，2=左，4=右，8=后
                          例如：1=仅前，3=前+左，7=前+左+右，15=所有麦克风
            frequency: 频率 (Hz)，默认16000
            buffer_size: 缓冲区大小
            sample_rate: 兼容参数，如果提供则覆盖frequency
            **kwargs: 额外参数

        Returns:
            int: 0表示成功，-1表示失败
        """
        # 日志记录参数
        self.logger.warning("启动音频流 - 客户端: %s, 通道掩码: %d (二进制: %s), 频率: %d", 
                           client_id, channels_mask, bin(channels_mask), frequency)
        
        # 处理兼容性参数
        effective_frequency = int(sample_rate) if sample_rate is not None else int(frequency)
        try:
            channels_mask = int(channels_mask)
        except (ValueError, TypeError):
            self.logger.error("无效的通道掩码值: %r", channels_mask)
            return -1
        
        # 确保NAOqi已初始化
        if not NAOQI_AVAILABLE:
            self.logger.error("无法启动音频流：NAOqi不可用")
            return -1
        
        if not hasattr(self, '_initialized') or not self._initialized:
            self.logger.warning("音频流初始化前先初始化NAOqi组件")
            if not self.initialize():
                self.logger.error("初始化NAOqi组件失败")
                return -1
        
        # 检查ZMQ套接字
        if not self.audio_stream_socket:
            self.logger.error("无法启动音频流：ZMQ套接字未配置")
            return -1
        
        # 检查是否已存在该客户端的流
        if client_id in self.streaming_clients:
            self.logger.warning("客户端 %s 的音频流已存在，先停止旧流", client_id)
            self.stop_audio_stream(client_id)
            # 短暂等待以确保旧流完全停止
            time.sleep(0.1)
        
        try:
            # 创建唯一的模块名称
            module_name = "AudioHandler_{0}".format(uuid.uuid4().hex[:8])
            self.logger.info("创建音频处理模块: %s", module_name)
            
            # 实例化处理器
            handler = AudioStreamHandler(
                module_name, 
                self.audio_stream_socket, 
                self.audio_send_lock, 
                client_id,
                self.logger
            )
            
            # 启用麦克风语音检测
            # 注意：这一步很重要，有些Pepper可能需要先启用麦克风
            try:
                # 尝试激活语音检测，但不要中断流程
                self.logger.info("尝试启用麦克风语音检测")
                if hasattr(self.audio_device, "setParameter"):
                    self.audio_device.setParameter("recordVadThreshold", 2.0)
                    self.audio_device.setParameter("recordNoiseReduction", 0.2)
                    self.logger.info("语音检测参数设置成功")
                else:
                    self.logger.warning("设备不支持设置语音检测参数")
            except Exception as vad_e:
                self.logger.warning("设置语音检测参数失败: %s", vad_e)
            
            # 注册回调
            self.logger.info("注册音频处理回调")
            try:
                # 1. 订阅前检查
                try:
                    subscribers_before = self.audio_device.getSubscribers()
                    self.logger.info("订阅前的订阅者列表: %s", str(subscribers_before))
                except Exception as e_check1:
                    self.logger.warning("获取订阅前列表失败: %s", e_check1)
                
                # 2. 设置客户端参数
                self.logger.info("设置客户端偏好 - 模块: %s, 频率: %d, 通道掩码: %d, 格式: 0", 
                                module_name, effective_frequency, channels_mask)
                self.audio_device.setClientPreferences(module_name, effective_frequency, channels_mask, 0)
                self.logger.info("客户端偏好设置成功")
                
                # 3. 订阅模块
                self.logger.info("订阅音频模块: %s", module_name)
                self.audio_device.subscribe(module_name)
                self.logger.info("音频回调订阅成功")
                
                # 4. 订阅后检查
                try:
                    subscribers_after = self.audio_device.getSubscribers()
                    self.logger.info("订阅后的订阅者列表: %s", str(subscribers_after))
                    if module_name not in subscribers_after:
                        self.logger.error("******** 严重错误：订阅成功但模块 %s 未出现在订阅者列表中！ ********", module_name)
                except Exception as e_check2:
                    self.logger.warning("获取订阅后列表失败: %s", e_check2)
                
            except Exception as sub_e:
                self.logger.error("设置客户端偏好或订阅音频回调失败: %s", sub_e, exc_info=True)
                # 尝试清理已创建的处理器，如果订阅失败
                try:
                    handler.stop()
                except:
                    pass
                return -1
            
            # 存储流信息
            self.streaming_clients[client_id] = {
                'handler': handler,
                'module_name': module_name
            }
            
            self.logger.warning("音频流已成功启动 - 客户端: %s, 模块: %s, 通道掩码: %d", 
                             client_id, module_name, channels_mask)
            return 0
            
        except Exception as e:
            self.logger.error("启动音频流时发生错误: %s", e, exc_info=True)
            return -1

    def stop_audio_stream(self, client_id):
        """停止特定客户端的音频流
        
        Args:
            client_id: 要停止音频流的客户端ID
            
        Returns:
            bool: 如果成功停止流或流本来就没在运行则返回True，否则返回False
        """
        self.logger.info("停止客户端 %s 的音频流", client_id)
        
        # 检查客户端是否在流
        if client_id not in self.streaming_clients:
            self.logger.warning("客户端 %s 没有活动的音频流", client_id)
            return True # 没有流，视为成功
        
        try:
            # 获取客户端信息
            client_info = self.streaming_clients[client_id]
            handler = client_info.get('handler')
            module_name = client_info.get('module_name')
            
            # 取消订阅并停止处理器
            if self.audio_device and module_name:
                try:
                    self.logger.info("正在取消订阅音频模块 %s", module_name)
                    self.audio_device.unsubscribe(module_name)
                    self.logger.info("取消订阅成功")
                except Exception as unsub_e:
                    self.logger.error("取消音频订阅时出错: %s", unsub_e)
                
            # 停止处理器
            if handler:
                try:
                    handler.stop()
                except Exception as stop_e:
                    self.logger.error("停止音频处理器时出错: %s", stop_e)
                
            # 移除客户端信息
            del self.streaming_clients[client_id]
            
            self.logger.warning("音频流已停止 - 客户端: %s, 剩余流: %d", 
                             client_id, len(self.streaming_clients))
            return True
            
        except Exception as e:
            self.logger.error("停止音频流时出错: %s", e, exc_info=True)
            return False

    def cleanup(self):
        """Clean up resources, stop all streams, and shut down the broker."""
        self.logger.info("Cleaning up SoundAgent...")

        # Stop all active streams
        active_client_ids = list(self.streaming_clients.keys())
        if active_client_ids:
            self.logger.info("Stopping %d active audio streams..." % len(active_client_ids))
            for client_id in active_client_ids:
                self.stop_audio_stream(client_id)
        else:
            self.logger.info("No active audio streams to stop.")

        # Shutdown the NAOqi broker if it exists
        if self.broker:
            try:
                broker_name = self.broker.getName()
                self.logger.info("Shutting down NAOqi broker '%s'..." % broker_name)
                self.broker.shutdown()
                self.logger.info("NAOqi broker '%s' shut down." % broker_name)
            except Exception as e:
                self.logger.error("Error shutting down NAOqi broker: %s" % e, exc_info=True)
            finally:
                self.broker = None
                self.audio_device = None # Clear proxy as well

        self.logger.info("SoundAgent cleanup finished.")

    def __del__(self):
        """Destructor to ensure cleanup is called."""
        self.cleanup()