#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import os
import sys
import platform
import socket
import urllib2
import traceback

class SystemAgent(object):
    """系统相关功能的代理
    
    提供系统状态、健康检查、版本信息等功能
    """
    
    def __init__(self, ip, port):
        """初始化系统代理
        
        Args:
            ip (str): 机器人IP
            port (int): NAOqi端口
        """
        self.ip = ip
        self.port = port
        self.start_time = time.time()
        self.debug = False
    
    def initialize(self):
        """初始化代理"""
        # 系统代理不需要连接到NAOqi服务，仅提供基本信息
        return True
    
    def set_debug(self, debug):
        """设置调试模式"""
        self.debug = debug
    
    def register_methods(self):
        """注册可用方法"""
        return {
            'system': {
                'ping': self.ping,
                'getServerInfo': self.get_server_info,
                'getUptime': self.get_uptime,
                'getVersion': self.get_version,
                'checkProxy': self.check_proxy,
                'testConnection': self.test_connection,
                'list_services': self.list_services,
                'list_methods': self.list_methods
            }
        }
    
    def ping(self):
        """健康检查的ping方法
        
        Returns:
            str: 固定返回"pong"字符串
        """
        return "pong"
    
    def get_server_info(self):
        """获取服务器信息
        
        Returns:
            dict: 包含服务器信息的字典
        """
        return {
            'hostname': platform.node(),
            'platform': platform.platform(),
            'python_version': sys.version,
            'uptime': self.get_uptime(),
            'startTime': time.strftime('%Y-%m-%d %H:%M:%S', 
                                       time.localtime(self.start_time))
        }
    
    def get_uptime(self):
        """获取服务器运行时间（秒）
        
        Returns:
            float: 服务器运行的秒数
        """
        return time.time() - self.start_time
    
    def get_version(self):
        """获取服务器版本
        
        Returns:
            str: 版本字符串
        """
        return "1.0.0"
        
    def check_proxy(self):
        """检查系统代理设置
        
        Returns:
            dict: 当前系统代理设置信息
        """
        proxy_info = {
            'http_proxy': os.environ.get('http_proxy', ''),
            'https_proxy': os.environ.get('https_proxy', ''),
            'no_proxy': os.environ.get('no_proxy', ''),
            'HTTP_PROXY': os.environ.get('HTTP_PROXY', ''),
            'HTTPS_PROXY': os.environ.get('HTTPS_PROXY', ''),
            'NO_PROXY': os.environ.get('NO_PROXY', '')
        }
        
        # 检查urllib是否使用了代理
        proxy_handler = urllib2.ProxyHandler({})
        opener = urllib2.build_opener(proxy_handler)
        urllib2.install_opener(opener)
        
        return proxy_info
        
    def test_connection(self, host="www.baidu.com", port=80, timeout=5):
        """测试网络连接
        
        Args:
            host (str): 要测试连接的主机
            port (int): 端口
            timeout (int): 超时时间（秒）
            
        Returns:
            dict: 连接测试结果
        """
        result = {
            'success': False,
            'error': None,
            'time_ms': 0
        }
        
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 创建socket连接
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            
            # 尝试连接
            s.connect((host, port))
            s.close()
            
            # 计算耗时
            end_time = time.time()
            result['time_ms'] = int((end_time - start_time) * 1000)
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            
        return result
    
    def list_services(self):
        """获取所有可用服务列表
        
        Returns:
            list: 可用服务列表
        """
        try:
            # 获取全局服务器实例中的服务列表
            import __main__
            if hasattr(__main__, 'server') and hasattr(__main__.server, '_method_map'):
                return list(__main__.server._method_map.keys())
            else:
                # 如果无法获取服务器实例，则返回一个基本的服务列表
                return ['system', 'tts', 'motion', 'sensor', 'behavior', 'speech']
        except Exception as e:
            if self.debug:
                print("获取服务列表异常: %s" % e)
                print(traceback.format_exc())
            # 出错时返回基础服务列表
            return ['system', 'tts', 'motion', 'sensor', 'behavior', 'speech']
    
    def list_methods(self, service_name=None):
        """获取指定服务或当前服务的方法列表
        
        Args:
            service_name (str, optional): 服务名称，如果为None则返回当前服务的方法列表
            
        Returns:
            list: 方法列表
        """
        try:
            # 设置默认服务名称
            if service_name is None:
                service_name = 'system'
                
            # 获取全局服务器实例中的方法列表
            import __main__
            if hasattr(__main__, 'server') and hasattr(__main__.server, '_method_map'):
                if service_name in __main__.server._method_map:
                    return list(__main__.server._method_map[service_name].keys())
                else:
                    return []
            else:
                # 如果无法获取服务器实例，则返回自己的方法列表
                if service_name == 'system':
                    return ['ping', 'getServerInfo', 'getUptime', 'getVersion', 
                            'checkProxy', 'testConnection', 'list_services', 'list_methods']
                else:
                    return []
        except Exception as e:
            if self.debug:
                print("获取方法列表异常: %s" % e)
                print(traceback.format_exc())
            # 出错时返回基础方法列表
            if service_name == 'system':
                return ['ping', 'getServerInfo', 'getUptime', 'getVersion', 
                        'checkProxy', 'testConnection', 'list_services', 'list_methods']
            else:
                return [] 