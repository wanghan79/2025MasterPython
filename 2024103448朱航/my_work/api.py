#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import zmq
import json
import logging
import time
import traceback
import whisper

class PepperRobotClient:
    """Pepper机器人客户端API
    
    提供与Pepper机器人服务器的ZMQ通信接口，支持以下功能模块：
    - 系统信息：获取电池状态、代理设置、网络连接测试
    - 动作控制：移动、转向、速度设置、唤醒/休息状态
    - 姿势控制：姿势切换、当前姿势获取
    - 自主生命：状态设置与查询
    - 行为管理：行为运行/停止、已安装行为列表
    - 语音交互：TTS语音合成、音量/语言设置
    - 传感器数据：电池电量、声纳距离等
    - 内存管理：数据存取、事件声明与触发
    """

    def __init__(self, host="localhost", port=5555, timeout=50000, 
                 retry_count=0, retry_interval=2.0, debug=False):
        """初始化客户端API
        ·
        Args:
            host (str): 服务器主机地址
            port (int): ZMQ服务端口
            timeout (int): 请求超时时间（毫秒）, 默认 20 秒
            retry_count (int): 连接失败时的重试次数 (注：内部重试已移除，此参数影响不大)
            retry_interval (float): 重试间隔时间（秒）(注：内部重试已移除)
            debug (bool): 是否启用调试模式
        """
        # 初始化日志
        self.logger = self._configure_logger(
            name="PepperClient", 
            debug=debug
        )
        self.debug = debug
        
        # 连接参数
        self.host = host
        self.port = port
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_interval = retry_interval
        self.endpoint = "tcp://%s:%s" % (host, port)
        
        # 设置ZMQ连接
        self.context = None
        self.socket = None
        self.audio_socket = None
        self._setup_connection()
        
    def _configure_logger(self, name, debug=False):
        """配置日志记录器"""
        logger = logging.getLogger(name)
        
        # 设置日志级别
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        
        # 如果没有处理器，添加一个
        if not logger.handlers:
            # 创建控制台处理器
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # 创建日志目录（如果不存在）
            log_dir = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), 'logs')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            # 添加文件处理器
            file_handler = logging.FileHandler(
                os.path.join(log_dir, 'pepper_client.log'),
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        return logger
        
    def _setup_connection(self):
        """建立ZMQ连接"""
        # 关闭现有连接（如果有）
        self._close_connection()
        
        # 创建新连接
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout)
        
        # 连接到服务器
        self.logger.info("连接到ZMQ服务器: %s", self.endpoint)
        self.socket.connect(self.endpoint)
    
    def _close_connection(self):
        """关闭ZMQ连接"""
        try:
            if self.socket:
                self.socket.close()
            if self.context:
                self.context.term()
        except Exception as e:
            self.logger.warning("关闭连接时出错: %s", e)
        finally:
            self.socket = None
            self.context = None
    
    def _send_request(self, service, method, args=None):
        """发送请求到服务器 (单次尝试)
        
        Args:
            service (str): 服务名称
            method (str): 方法名称
            args (list): 方法参数列表
            
        Returns:
            返回服务器响应结果
            
        Raises:
            Exception: 如果请求失败（包括超时、连接错误等），则抛出异常
        """
        if args is None:
            args = []
            
        request = {
            "service": service,
            "method": method,
            "args": args
        }
        
        # 确保连接已建立 (如果需要)
        # 可以考虑在这里添加: if not self.socket: self._setup_connection()
        # 但更推荐在 __init__ 中确保连接成功，或让调用者处理连接失败
        if not self.socket or not self.context:
             # 如果没有活动的连接，则尝试建立连接
             try:
                 self._setup_connection()
             except Exception as setup_e:
                 self.logger.error("发送请求前建立连接失败: %s", setup_e)
                 raise ConnectionError("无法建立 ZMQ 连接") from setup_e

        try:
            # 转换请求为JSON字符串
            request_str = json.dumps(request)
            if self.debug:
                self.logger.debug("发送请求: %s", request_str)
            
            # 发送请求
            # 注意：REQ 套接字发送后必须等待接收
            self.socket.send_string(request_str)
            
            # 接收响应 (会根据 RCVTIMEO 超时)
            response_str = self.socket.recv_string()
            response = json.loads(response_str)
            
            if self.debug:
                self.logger.debug("接收响应: %s", response)
            
            # 检查响应状态
            if response.get("status") == "error":
                error_msg = response.get("error", "未知服务器错误")
                self.logger.error("请求失败: %s", error_msg)
                # 可以考虑抛出一个更具体的自定义异常
                raise Exception(error_msg) 
            
            # 返回结果
            return response.get("result")
            
        except zmq.Again as e:
            # 超时错误
            error_msg = "请求超时 (超过 %d ms)" % self.timeout
            self.logger.error(error_msg)
            # 超时通常不关闭连接，但需要通知调用者
            raise TimeoutError(error_msg) from e
            
        except zmq.ZMQError as e:
            # 其他 ZMQ 相关错误，可能表示连接问题
            error_msg = "ZMQ 通信错误: %s" % str(e)
            self.logger.error(error_msg)
            # 关闭可能损坏的连接，以便下次尝试重新建立
            self._close_connection()
            raise ConnectionError(error_msg) from e
            
        except Exception as e:
            # 其他所有错误 (如 JSON 解析错误等)
            error_msg = "处理请求或响应时发生错误: %s" % str(e)
            self.logger.error(error_msg)
            self.logger.error("异常堆栈: %s", traceback.format_exc())
            # 对于未知错误，也可以考虑关闭连接
            # self._close_connection()
            raise # 直接重新抛出原始异常
    
    def check_connection(self):
        """检查与服务器的连接是否正常
        
        Returns:
            bool: 连接是否正常
        """
        try:
            result = self._send_request("system", "ping", [])
            return result == "pong"
        except Exception as e:
            self.logger.error("连接检查失败: %s", e)
            return False

    def check_robot_connection(self, host="www.baidu.com", port=80, timeout=5):
        """测试机器人的网络连接
        
        Args:
            host (str): 测试连接的目标主机
            port (int): 目标端口
            timeout (int): 超时时间（秒）
            
        Returns:
            dict: 连接测试结果
        """
        try:
            return self._send_request("system", "testConnection", [host, port, timeout])
        except Exception as e:
            self.logger.error("测试连接失败: %s", e)
            return {
                'success': False,
                'error': str(e),
                'time_ms': 0
            }
            
    def check_robot_proxy(self):
        """检查机器人的代理设置
        
        Returns:
            dict: 代理设置信息
        """
        try:
            return self._send_request("system", "checkProxy", [])
        except Exception as e:
            self.logger.error("检查代理设置失败: %s", e)
            return {}
    
    def close(self):
        """关闭连接并清理资源"""
        try:
            self._close_connection()
            if self.audio_socket:
                self.audio_socket.close()
                self.audio_socket = None
            self.logger.info("API连接已关闭")
        except Exception as e:
            self.logger.warning("关闭API连接时发生错误: %s", e)
            
    def start_receiving_audio(self, port=5556):
        """开始接收音频流
        
        Args:
            port (int): 音频流端口号
        """
        try:
            if not self.context:
                self.context = zmq.Context()
                
            self.audio_socket = self.context.socket(zmq.SUB)
            self.audio_socket.setsockopt(zmq.LINGER, 0)
            self.audio_socket.setsockopt(zmq.SUBSCRIBE, b'')
            self.audio_socket.connect(f"tcp://{self.host}:{port}")
            self.logger.info("已连接到音频流端口: %s", port)
            return True
        except Exception as e:
            self.logger.error("连接音频流失败: %s", e)
            return False
            
    def stop_receiving_audio(self):
        """停止接收音频流"""
        if self.audio_socket:
            self.audio_socket.close()
            self.audio_socket = None
            self.logger.info("已断开音频流连接")
            return True
        return False
        
    def get_next_audio_chunk(self, timeout_ms=None):
        """获取下一个音频数据块
        
        Args:
            timeout_ms (int): 超时时间(毫秒)
            
        Returns:
            bytes: 音频数据块，超时返回None
        """
        if not self.audio_socket:
            self.logger.warning("音频流未初始化")
            return None
            
        try:
            if timeout_ms is not None:
                self.audio_socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
                
            return self.audio_socket.recv()
        except zmq.Again:
            return None
        except Exception as e:
            self.logger.error("接收音频数据时出错: %s", e)
            return None

    # ========== 系统接口 ==========
    
    def get_system_info(self):
        """获取系统信息
        
        返回:
            dict: 系统信息字典，包含以下字段：
                - battery: 电池电量
                - behaviors: 已安装的行为列表
                - posture: 当前姿势
                - life_state: 自主生命状态
        """
        try:
            info = {}
            
            # 获取电池电量
            battery = self._send_request("memory", "get_data", ["Device/SubDeviceList/Battery/Charge/Sensor/Value"])
            if battery is not None:
                info['battery'] = battery
            else:
                self.logger.warning("无法获取电池电量信息")
            
            # 获取已安装的行为
            try:
                behaviors = self._send_request("behavior", "get_installed_behaviors")
                if behaviors:
                    info['behaviors'] = behaviors
            except Exception as e:
                self.logger.warning("无法获取已安装的行为列表: %s", str(e))
            
            # 获取机器人状态
            try:
                posture = self._send_request("posture", "get_posture")
                if posture:
                    info['posture'] = posture
            except Exception as e:
                self.logger.warning("无法获取机器人姿势: %s", str(e))
            
            # 获取自主生命状态
            try:
                life_state = self._send_request("life", "get_autonomous_life_state")
                if life_state:
                    info['life_state'] = life_state
            except Exception as e:
                self.logger.warning("无法获取自主生命状态: %s", str(e))
            
            return info
        except Exception as e:
            self.logger.error("获取系统信息失败: %s", str(e))
            self.logger.error("异常堆栈: %s", traceback.format_exc())
            return {}

    # ========== 动作控制接口 ==========
    
    def move_to(self, x, y, theta):
        """移动到指定位置

        参数:
            x (float): X轴距离（米）
            y (float): Y轴距离（米）
            theta (float): 旋转角度（弧度）

        返回:
            bool: 是否成功
        """
        return self._send_request("motion", "moveTo", [x, y, theta])
    
    def stop_move(self):
        """停止移动

        返回:
            bool: 是否成功
        """
        return self._send_request("motion", "stopMove")
    
    def set_velocity(self, x, y, theta):
        """设置移动速度

        参数:
            x (float): X轴速度（-1.0到1.0）
            y (float): Y轴速度（-1.0到1.0）
            theta (float): 旋转速度（-1.0到1.0）

        返回:
            bool: 是否成功
        """
        return self._send_request("motion", "setVelocity", [x, y, theta])
    
    def wake_up(self):
        """唤醒机器人（启用电机）

        返回:
            bool: 是否成功
        """
        return self._send_request("motion", "wakeUp")
    
    def rest(self):
        """休息状态（禁用电机）

        返回:
            bool: 是否成功
        """
        return self._send_request("motion", "rest")

    # ========== 姿势控制接口 ==========
    
    def go_to_posture(self, posture_name, speed=0.5):
        """设置机器人姿势

        参数:
            posture_name (str): 姿势名称，如'Stand'、'Crouch'等
            speed (float): 速度，范围0.0-1.0

        返回:
            bool: 是否成功
        """
        return self._send_request("posture", "go_to_posture", [posture_name, speed])
    
    def get_posture(self):
        """获取当前姿势

        返回:
            str: 当前姿势名称
        """
        return self._send_request("posture", "get_posture")

    # ========== 自主生命接口 ==========
    
    def set_autonomous_life_state(self, state):
        """设置自主生命状态

        参数:
            state (str): 状态，可选值为'solitary'、'interactive'、'disabled'等

        返回:
            bool: 是否成功
        """
        return self._send_request("life", "set_autonomous_life_state", [state])
    
    def get_autonomous_life_state(self):
        """获取自主生命状态

        返回:
            str: 当前状态
        """
        return self._send_request("life", "get_autonomous_life_state")

    # ========== 行为控制接口 ==========
    
    def run_behavior(self, behavior_name):
        """运行行为

        参数:
            behavior_name (str): 行为名称

        返回:
            bool: 是否成功
        """
        return self._send_request("behavior", "run_behavior", [behavior_name])
    
    def stop_behavior(self, behavior_name):
        """停止行为

        参数:
            behavior_name (str): 行为名称

        返回:
            bool: 是否成功
        """
        return self._send_request("behavior", "stop_behavior", [behavior_name])
    
    def stop_all_behaviors(self):
        """停止所有行为

        返回:
            bool: 是否成功
        """
        return self._send_request("behavior", "stop_all_behaviors")
    
    def get_installed_behaviors(self):
        """获取已安装的行为列表

        返回:
            list: 行为名称列表
        """
        return self._send_request("behavior", "get_installed_behaviors")

    # ========== TTS接口 ==========
    
    def say(self, text):
        """让机器人说话

        参数:
            text (str): 要说的文本

        返回:
            bool: 是否成功
        """
        return self._send_request("tts", "say", [text])
    
    def set_language(self, language):
        """设置TTS语言

        参数:
            language (str): 语言代码 (例如: "Chinese", "English")

        返回:
            bool: 是否成功
        """
        return self._send_request("tts", "set_language", [language])
    
    def set_volume(self, volume):
        """设置TTS音量

        参数:
            volume (float): 音量 (0.0-1.0)

        返回:
            bool: 是否成功
        """
        return self._send_request("tts", "set_volume", [volume])
    
    def set_tts_parameter(self, param, value):
        """设置TTS参数

        参数:
            param (str): 参数名称 (例如: "speed", "pitch")
            value: 参数值

        返回:
            bool: 是否成功
        """
        return self._send_request("tts", "set_parameter", [param, value])

  

    # ========== 传感器接口 ==========
    
    def get_battery_level(self):
        """获取电池电量
        
        返回:
            float: 电池电量百分比
        """
        return self._send_request("sensor", "get_battery_level")
    
    def get_sonar_value(self, sonar_id=0):
        """获取声纳传感器值
        
        参数:
            sonar_id (int): 声纳ID，0为前置，1为后置
            
        返回:
            float: 声纳距离值（米）
        """
        return self._send_request("sensor", "get_sonar_value", [sonar_id])

    # ========== 内存数据接口 ==========
    
    def get_memory_data(self, key):
        """获取内存中的数据

        参数:
            key (str): 数据键名

        返回:
            数据值，如果不存在则返回None
        """
        return self._send_request("memory", "get_data", [key])
    
    def get_memory_data_list(self, pattern):
        """获取匹配模式的所有数据键名

        参数:
            pattern (str): 匹配模式

        返回:
            list: 匹配的数据键名列表
        """
        return self._send_request("memory", "get_data_list", [pattern])
    
    def insert_memory_data(self, key, value):
        """插入数据到内存

        参数:
            key (str): 数据键名
            value: 数据值

        返回:
            bool: 是否成功
        """
        return self._send_request("memory", "insert_data", [key, value])
    
    def remove_memory_data(self, key):
        """从内存中删除数据

        参数:
            key (str): 数据键名

        返回:
            bool: 是否成功
        """
        return self._send_request("memory", "remove_data", [key])
    
    def declare_memory_event(self, event_name):
        """声明一个事件

        参数:
            event_name (str): 事件名称

        返回:
            bool: 是否成功
        """
        return self._send_request("memory", "declare_event", [event_name])
    
    def raise_memory_event(self, event_name, value):
        """触发一个事件

        参数:
            event_name (str): 事件名称
            value: 事件值

        返回:
            bool: 是否成功
        """
        return self._send_request("memory", "raise_event", [event_name, value])
    
    def get_event_history(self, event_name, max_size=100):
        """获取事件历史记录

        参数:
            event_name (str): 事件名称
            max_size (int): 最大记录数

        返回:
            list: 事件历史记录列表
        """
        return self._send_request("memory", "get_event_history", [event_name, max_size])
    
    def subscribe_to_memory_event(self, event_name, callback):
        """订阅事件

        参数:
            event_name (str): 事件名称
            callback: 回调函数

        返回:
            bool: 是否成功
        """
        return self._send_request("memory", "subscribe_to_event", [event_name, callback])
    
    def unsubscribe_to_memory_event(self, event_name):
        """取消订阅事件

        参数:
            event_name (str): 事件名称

        返回:
            bool: 是否成功
        """
        return self._send_request("memory", "unsubscribe_to_event", [event_name])

    # ========== 平板控制接口 ==========
    
    def show_image(self, image_path):
        """在平板上显示图片
        
        参数:
            image_path (str): 图片文件路径
            
        返回:
            bool: 操作是否成功
        """
        return self._send_request("tablet", "show_image", [image_path])
    
    def show_web_page(self, url):
        """在平板上显示网页
        
        参数:
            url (str): 网页URL
            
        返回:
            bool: 操作是否成功
        """
        return self._send_request("tablet", "show_web_page", [url])
    
    def hide_web_page(self):
        """隐藏当前显示的网页
        
        返回:
            bool: 操作是否成功
        """
        return self._send_request("tablet", "hide_web_page")
    
    def execute_js(self, js_code):
        """在平板当前网页中执行JavaScript代码
        
        参数:
            js_code (str): JavaScript代码
            
        返回:
            bool: 操作是否成功
        """
        return self._send_request("tablet", "execute_js", [js_code])
    
    def reset_tablet(self):
        """重置平板显示
        
        返回:
            bool: 操作是否成功
        """
        return self._send_request("tablet", "reset_tablet")
    
    def show_emoji(self, emoji_name, title=None, say_name=False):
        """在平板上显示SVG表情
        
        使用可靠的SVG表情系统显示表情符号，解决了传统图片文件在Pepper平板上难以正常显示的问题。
        SVG表情具有以下优势：
        1. 无需外部文件依赖，不受文件访问限制
        2. 矢量图形，可任意缩放而不失真
        3. 轻量级，加载速度快
        4. 可通过代码动态生成和修改
        
        可用的表情包括：
        - smile: 笑脸表情
        - sad: 伤心表情
        - surprise: 惊讶表情
        - love: 爱心表情
        - think: 思考表情
        
        使用示例：
            robot.show_emoji("smile")               # 显示笑脸
            robot.show_emoji("sad", say_name=True)  # 显示伤心表情并说出名称
            robot.show_emoji("surprise", title="哇！惊喜！")  # 显示惊讶表情并自定义标题
        
        参数:
            emoji_name (str): 表情名称，可以是'smile', 'sad', 'surprise', 'love', 'think'
            title (str, optional): 自定义标题，如果为None则使用默认标题
            say_name (bool, optional): 是否同时说出表情名称，默认为False
            
        返回:
            bool: 操作是否成功
        """
        return self._send_request("tablet", "show_emoji", [emoji_name, title, say_name])
    
    def get_available_emojis(self):
        """获取所有可用的表情名称列表
        
        Returns:
            list: 表情名称列表
        """
        return self._send_request("tablet", "get_available_emojis", [])
    
    # 声音录制和传输功能
    def set_recordings_dir(self, dir_path):
        """设置机器人上存储录音的目录
        
        Args:
            dir_path (str): 录音存储路径（使用正斜杠）
            
        Returns:
            bool: 是否成功
        """
        return self._send_request("sound", "set_recordings_dir", [dir_path])
    
    def get_recordings_dir(self):
        """获取机器人上存储录音的目录
        
        Returns:
            str: 录音存储路径
        """
        return self._send_request("sound", "get_recordings_dir", [])
    
    def start_recording(self, filename, format_type='wav', samplerate=16000, channels=(1, 1, 1, 1)):
        """开始录音
        
        Args:
            filename (str): 保存录音的路径（使用正斜杠）
            format_type (str, optional): 文件格式，默认wav
            samplerate (int, optional): 采样率，默认16000
            channels (tuple, optional): 通道设置，格式（前，后，左，右）
            
        Returns:
            bool: 是否成功
        """
        return self._send_request("sound", "start_recording", [filename, format_type, samplerate, channels])
    
    def post_start_recording(self, filename, format_type='wav', samplerate=16000, channels=(1, 1, 1, 1)):
        """异步开始录音
        
        Args:
            filename (str): 保存录音的路径（使用正斜杠）
            format_type (str, optional): 文件格式，默认wav
            samplerate (int, optional): 采样率，默认16000
            channels (tuple, optional): 通道设置，格式（前，后，左，右）
            
        Returns:
            bool: 是否成功
        """
        return self._send_request("sound", "post_start_recording", [filename, format_type, samplerate, channels])
    
    def stop_recording(self):
        """停止录音
        
        Returns:
            bool: 是否成功
        """
        return self._send_request("sound", "stop_recording", [])
    
    def record_audio(self, filename, duration, format_type='wav', samplerate=16000, channels=(1, 0, 0, 0)):
        """录制指定时长的音频
        
        Args:
            filename (str): 文件名（不含路径）
            duration (float): 录音时长（秒）
            format_type (str, optional): 文件格式，默认wav
            samplerate (int, optional): 采样率，默认16000
            channels (tuple, optional): 通道设置，格式（前，后，左，右）
            
        Returns:
            tuple: (是否成功, 机器人上录音的完整路径)
        """
        return self._send_request("sound", "record_audio", [filename, duration, format_type, samplerate, channels])
    
    def transfer_file(self, robot_file, local_dir, username='nao', password='pepper2023'):
        """使用SCP方式将文件从机器人传输到计算机
        
        Args:
            robot_file (str): 机器人上的文件路径
            local_dir (str): 本地保存目录
            username (str, optional): 机器人用户名
            password (str, optional): 机器人密码
            
        Returns:
            tuple: (是否成功, 本地保存的文件路径或错误信息)
        """
        return self._send_request("sound", "transfer_file", [robot_file, local_dir, username, password])
    
    def transfer_file_scp(self, robot_file, local_dir, username='nao', password='pepper2023'):
        """使用SCP将文件从机器人传输到计算机（需要paramiko库）
        
        Args:
            robot_file (str): 机器人上的文件路径
            local_dir (str): 本地保存目录
            username (str, optional): 机器人SSH用户名
            password (str, optional): 机器人SSH密码
            
        Returns:
            tuple: (是否成功, 本地保存的文件路径或错误信息)
        """
        return self._send_request("sound", "transfer_file_scp", [robot_file, local_dir, username, password])
    
    def record_and_transfer(self, filename, duration, local_dir, 
                           format_type='wav', samplerate=16000, channels=(1, 0, 0, 0),
                           username='nao', password='pepper2023'):
        """录制音频并通过SCP传输到电脑
        
        对于长时间录音（超过3秒），建议增加超时时间设置
        
        Args:
            filename (str): 文件名（不含路径）
            duration (float): 录制时长（秒）
            local_dir (str): 本地保存目录
            format_type (str, optional): 文件格式，默认wav
            samplerate (int, optional): 采样率，默认16000
            channels (tuple, optional): 通道设置，格式为(前，后，左，右)
            username (str, optional): 机器人SSH用户名
            password (str, optional): 机器人SSH密码
            
        Returns:
            bool: 成功或失败
            str: 本地保存的文件路径或错误信息
        """
        # 计算预期超时时间 - 录音时长 + 传输时间（估计为5秒）+ 额外缓冲（1秒）
        expected_time = duration + 5.0 + 1.0
        current_timeout = self.timeout
        
        # 如果当前超时设置小于预期时间，则临时增加超时
        temp_timeout_change = False
        if (current_timeout / 1000.0) < expected_time:
            temp_timeout_change = True
            new_timeout = int(expected_time * 1000) + 5000  # 转为毫秒并额外增加5秒缓冲
            self.logger.info("临时增加超时时间从 %d 毫秒到 %d 毫秒用于录音", current_timeout, new_timeout)
            # 保存原有超时设置，稍后恢复
            old_timeout = current_timeout
            # 设置新的超时
            self.timeout = new_timeout
            try:
                # 重新配置Socket超时
                if self.socket:
                    self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)
                    self.socket.setsockopt(zmq.SNDTIMEO, self.timeout)
            except Exception as e:
                self.logger.warning("设置超时失败: %s", e)
        
        try:
            # 调用服务器录制音频并传输
            result = self._send_request("sound", "record_and_transfer", 
                       [filename, duration, local_dir, format_type, samplerate, 
                        channels, username, password])
            
            # 结果格式：[success, detail]
            success = result[0] if isinstance(result, list) and len(result) > 0 else False
            detail = result[1] if isinstance(result, list) and len(result) > 1 else "未知结果"
            return success, detail
        
        except Exception as e:
            self.logger.error("录制和传输出错: %s", e)
            return False, "录制和传输异常: " + str(e)
            
        finally:
            # 如果临时修改了超时，恢复原有设置
            if temp_timeout_change:
                self.logger.info("恢复原有超时设置: %d 毫秒", old_timeout)
                self.timeout = old_timeout
                try:
                    # 重新配置Socket超时
                    if self.socket:
                        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)
                        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout)
                except Exception as e:
                    self.logger.warning("恢复超时设置失败: %s", e)