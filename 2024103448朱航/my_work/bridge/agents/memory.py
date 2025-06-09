#!/usr/bin/env python
# -*- coding: utf-8 -*-

from naoqi import ALProxy
from . import BaseAgent
import traceback

class MemoryAgent(BaseAgent):
    """Pepper机器人内存数据访问代理"""
    
    def __init__(self, ip, port):
        """初始化内存数据访问代理
        
        Args:
            ip (str): 机器人IP地址
            port (int): NAOqi端口号
        """
        super(MemoryAgent, self).__init__(ip, port)
        
    def _create_service_proxy(self):
        """创建内存数据访问服务代理"""
        self._proxies['ALMemory'] = ALProxy("ALMemory", self.ip, self.port)

    def register_methods(self):
        """注册供客户端调用的方法"""
        return {
            "memory": {
                "get_data": self.get_data,
                "get_data_list": self.get_data_list,
                "insert_data": self.insert_data,
                "remove_data": self.remove_data,
                "declare_event": self.declare_event,
                "raise_event": self.raise_event,
                "get_event_history": self.get_event_history,
                "subscribe_to_event": self.subscribe_to_event,
                "unsubscribe_to_event": self.unsubscribe_to_event
            }
        }
    
    def get_memory_proxy(self):
        """获取内存数据访问代理"""
        return self.get_proxy("ALMemory")
    
    def get_data(self, key):
        """获取内存中的数据
        
        Args:
            key (str): 数据键名
            
        Returns:
            数据值，如果不存在则返回None
        """
        try:
            memory_proxy = self.get_memory_proxy()
            value = memory_proxy.getData(key)
            
            if self._debug:
                print("获取数据: %s = %s" % (key, value))
            return value
        except Exception as e:
            if self._debug:
                print("获取数据失败: %s" % e)
                print(traceback.format_exc())
            return None
            
    def get_data_list(self, pattern):
        """获取匹配模式的所有数据键名
        
        Args:
            pattern (str): 匹配模式
            
        Returns:
            list: 匹配的数据键名列表
        """
        try:
            memory_proxy = self.get_memory_proxy()
            keys = memory_proxy.getDataList(pattern)
            
            if self._debug:
                print("获取数据列表: %s" % keys)
            return keys
        except Exception as e:
            if self._debug:
                print("获取数据列表失败: %s" % e)
                print(traceback.format_exc())
            return []
            
    def insert_data(self, key, value):
        """插入数据到内存
        
        Args:
            key (str): 数据键名
            value: 数据值
            
        Returns:
            bool: 是否成功
        """
        try:
            memory_proxy = self.get_memory_proxy()
            memory_proxy.insertData(key, value)
            
            if self._debug:
                print("插入数据: %s = %s" % (key, value))
            return True
        except Exception as e:
            if self._debug:
                print("插入数据失败: %s" % e)
                print(traceback.format_exc())
            return False
            
    def remove_data(self, key):
        """从内存中删除数据
        
        Args:
            key (str): 数据键名
            
        Returns:
            bool: 是否成功
        """
        try:
            memory_proxy = self.get_memory_proxy()
            memory_proxy.removeData(key)
            
            if self._debug:
                print("删除数据: %s" % key)
            return True
        except Exception as e:
            if self._debug:
                print("删除数据失败: %s" % e)
                print(traceback.format_exc())
            return False
            
    def declare_event(self, event_name):
        """声明一个事件
        
        Args:
            event_name (str): 事件名称
            
        Returns:
            bool: 是否成功
        """
        try:
            memory_proxy = self.get_memory_proxy()
            memory_proxy.declareEvent(event_name)
            
            if self._debug:
                print("声明事件: %s" % event_name)
            return True
        except Exception as e:
            if self._debug:
                print("声明事件失败: %s" % e)
                print(traceback.format_exc())
            return False
            
    def raise_event(self, event_name, value):
        """触发一个事件
        
        Args:
            event_name (str): 事件名称
            value: 事件值
            
        Returns:
            bool: 是否成功
        """
        try:
            memory_proxy = self.get_memory_proxy()
            memory_proxy.raiseEvent(event_name, value)
            
            if self._debug:
                print("触发事件: %s = %s" % (event_name, value))
            return True
        except Exception as e:
            if self._debug:
                print("触发事件失败: %s" % e)
                print(traceback.format_exc())
            return False
            
    def get_event_history(self, event_name, max_size=100):
        """获取事件历史记录
        
        Args:
            event_name (str): 事件名称
            max_size (int): 最大记录数
            
        Returns:
            list: 事件历史记录列表
        """
        try:
            memory_proxy = self.get_memory_proxy()
            history = memory_proxy.getEventHistory(event_name, max_size)
            
            if self._debug:
                print("获取事件历史: %s" % history)
            return history
        except Exception as e:
            if self._debug:
                print("获取事件历史失败: %s" % e)
                print(traceback.format_exc())
            return []
            
    def subscribe_to_event(self, event_name, callback):
        """订阅事件
        
        Args:
            event_name (str): 事件名称
            callback: 回调函数
            
        Returns:
            bool: 是否成功
        """
        try:
            memory_proxy = self.get_memory_proxy()
            memory_proxy.subscribeToEvent(event_name, "PepperBridge", callback)
            
            if self._debug:
                print("订阅事件: %s" % event_name)
            return True
        except Exception as e:
            if self._debug:
                print("订阅事件失败: %s" % e)
                print(traceback.format_exc())
            return False
            
    def unsubscribe_to_event(self, event_name):
        """取消订阅事件
        
        Args:
            event_name (str): 事件名称
            
        Returns:
            bool: 是否成功
        """
        try:
            memory_proxy = self.get_memory_proxy()
            memory_proxy.unsubscribeToEvent(event_name, "PepperBridge")
            
            if self._debug:
                print("取消订阅事件: %s" % event_name)
            return True
        except Exception as e:
            if self._debug:
                print("取消订阅事件失败: %s" % e)
                print(traceback.format_exc())
            return False 