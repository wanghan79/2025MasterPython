#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pepper机器人代理模块
包含各种控制机器人的代理类
"""

from naoqi import ALProxy
import traceback
import time

from abc import ABCMeta, abstractmethod

class BaseAgent(object):
    __metaclass__ = ABCMeta
    """所有代理的基类"""
    
    def __init__(self, ip, port):
        """初始化基础代理
        
        Args:
            ip (str): 机器人IP地址
            port (int): NAOqi端口号
        """
        self.ip = ip
        self.port = port
        self._debug = False
        self._proxies = {}  # 存储所有NAOqi代理
        
    def initialize(self):
        """初始化代理，创建服务连接"""
        self._create_service_proxy()
    
    @abstractmethod
    def _create_service_proxy(self):
        """抽象方法：创建服务代理，由子类实现具体逻辑"""
        pass
        
    def set_debug(self, debug):
        """设置调试模式
        
        Args:
            debug (bool): 是否启用调试模式
        """
        self._debug = debug
        
    def get_proxy(self, service_name=None):
        """获取服务代理
        
        Args:
            service_name (str, optional): 服务名称，如果提供则返回指定服务的代理
            
        Returns:
            ALProxy: 服务代理对象
        """
        if service_name is None:
            # 原来的行为：返回第一个代理
            return next(iter(self._proxies.values()))
        elif service_name in self._proxies:
            # 新行为：返回指定名称的代理
            return self._proxies[service_name]
        else:
            # 服务不存在
            raise Exception("服务 %s 未初始化" % service_name)
    
    def cleanup(self):
        """清理资源"""
        self._proxies.clear()
        
    def get_available_services(self):
        """获取可用服务列表
        
        Returns:
            list: 可用服务列表
        """
        return list(self._proxies.keys())

from .motion import MotionAgent
from .speech import SpeechAgent
from .vision import VisionAgent
from .behavior import BehaviorAgent
from .sensor import SensorAgent
from .tts import TTSAgent
from .memory import MemoryAgent
from .tablet import TabletAgent
from .system import SystemAgent
from .sound import SoundAgent
    
__all__ = [
    'MotionAgent',
    'SpeechAgent',
    'VisionAgent',
    'BehaviorAgent',
    'SensorAgent',
    'TTSAgent',
    'MemoryAgent',
    'TabletAgent',
    'SystemAgent',
    'SoundAgent'
]