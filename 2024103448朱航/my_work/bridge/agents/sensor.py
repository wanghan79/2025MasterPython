#!/usr/bin/env python
# -*- coding: utf-8 -*-

from naoqi import ALProxy
from . import BaseAgent
import traceback
import time

class SensorAgent(BaseAgent):
    """Pepper机器人传感器控制代理"""
    
    def register_methods(self):
        """注册传感器服务相关方法"""
        return {
            'sensor': {
                'get_touch_data': self.get_touch_status,
                'get_battery_level': self.get_battery_level,
                'get_sonar_value': self.get_sonar_distance
            },
            'memory': {
                'getData': self.get_data
            }
        }

    def __init__(self, ip, port):
        """初始化传感器控制代理
        
        Args:
            ip (str): 机器人IP地址
            port (int): NAOqi端口
        """
        super(SensorAgent, self).__init__(ip, port)
        
    def _create_service_proxy(self):
        """创建传感器服务代理"""
        self._proxies['ALMemory'] = ALProxy("ALMemory", self.ip, self.port)
        self._proxies['ALTouch'] = ALProxy("ALTouch", self.ip, self.port)
        self._proxies['ALSonar'] = ALProxy("ALSonar", self.ip, self.port)
        
    def _get_memory_proxy(self):
        """获取ALMemory代理"""
        return self.get_proxy("ALMemory")
    
    def _get_touch_proxy(self):
        """获取ALTouch代理"""
        return self.get_proxy("ALTouch")
    
    def _get_sonar_proxy(self):
        """获取ALSonar代理"""
        return self.get_proxy("ALSonar")
    
    def get_data(self, key):
        """获取内存数据
        
        Args:
            key (str): 数据键名
            
        Returns:
            any: 数据
        """
        try:
            memory_proxy = self._get_memory_proxy()
            
            if self._debug:
                print("获取内存数据: %s" % key)
            value = memory_proxy.getData(key)
            return value
        except Exception as e:
            if self._debug:
                print("获取内存数据异常: %s" % e)
                print(traceback.format_exc())
            return None
    
    def set_data(self, key, value):
        """设置内存数据
        
        Args:
            key (str): 数据键名
            value: 数据
            
        Returns:
            bool: 操作是否成功
        """
        try:
            memory_proxy = self._get_memory_proxy()
            
            if self._debug:
                print("设置内存数据: %s = %s" % (key, value))
            memory_proxy.insertData(key, value)
            return True
        except Exception as e:
            if self._debug:
                print("设置内存数据异常: %s" % e)
                print(traceback.format_exc())
            return False
    
    def subscribe_to_event(self, event, callback):
        """订阅事件
        
        Args:
            event (str): 事件名称
            callback (function): 回调函数
            
        Returns:
            bool: 操作是否成功
        """
        try:
            memory_proxy = self._get_memory_proxy()
            
            if self._debug:
                print("订阅事件: %s" % event)
            memory_proxy.subscribeToEvent(event, callback)
            return True
        except Exception as e:
            if self._debug:
                print("订阅事件异常: %s" % e)
                print(traceback.format_exc())
            return False
    
    def unsubscribe_from_event(self, event):
        """取消订阅事件
        
        Args:
            event (str): 事件名称
            
        Returns:
            bool: 操作是否成功
        """
        try:
            memory_proxy = self._get_memory_proxy()
            
            if self._debug:
                print("取消订阅事件: %s" % event)
            memory_proxy.unsubscribeToEvent(event)
            return True
        except Exception as e:
            if self._debug:
                print("取消订阅事件异常: %s" % e)
                print(traceback.format_exc())
            return False
    
    def get_touch_status(self):
        """获取触摸传感器状态
        
        Returns:
            dict: 触摸传感器状态
        """
        try:
            touch_proxy = self._get_touch_proxy()
            memory_proxy = self._get_memory_proxy()
            
            if self._debug:
                print("获取触摸传感器状态")
            
            # 获取所有触摸传感器
            sensors = touch_proxy.getSensorList()
            
            # 获取每个传感器的状
            result = {}
            for sensor in sensors:
                try:
                    value = memory_proxy.getData("Device/SubDeviceList/%s/Value" % sensor)
                    result[sensor] = value
                except:
                    pass
            
            return result
        except Exception as e:
            if self._debug:
                print("获取触摸传感器状态异常 %s" % e)
                print(traceback.format_exc())
            return {}
    
    def get_sonar_distance(self, sonar_id=0):
        """获取声纳距离
        
        参数:
            sonar_id (int): 声纳ID，0为前置，1为后置
            
        返回:
            float: 声纳距离值（米）
        """
        try:
            memory_proxy = self._get_memory_proxy()
            
            if self._debug:
                print("获取声纳距离")
            
            # 根据sonar_id选择对应的声纳
            if sonar_id == 0:
                key = "Device/SubDeviceList/US/Front/Sensor/Value"
            else:
                key = "Device/SubDeviceList/US/Back/Sensor/Value"
            
            # 获取声纳数据
            distance = memory_proxy.getData(key)
            return distance
        except Exception as e:
            if self._debug:
                print("获取声纳距离异常: %s" % e)
                print(traceback.format_exc())
            return 0
    
    def get_battery_level(self):
        """获取电池电量
        
        Returns:
            float: 电池电量百分比
        """
        try:
            memory_proxy = self._get_memory_proxy()
            
            if self._debug:
                print("获取电池电量")
            level = memory_proxy.getData("Device/SubDeviceList/Battery/Charge/Sensor/Value")
            return level * 100  # 转换为百分比
        except Exception as e:
            if self._debug:
                print("获取电池电量异常: %s" % e)
                print(traceback.format_exc())
            return 0
