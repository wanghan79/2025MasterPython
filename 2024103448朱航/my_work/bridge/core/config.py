#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import sys

# 设置默认编码为UTF-8
if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding('utf-8')

DEFAULT_CONFIG = {
    'robot': {
        'ip': '192.168.1.119',
        'naoqi_port': 9559,
        'zmq_port': 5555
    },
    'services': {
        'motion': True,
        'tts': True,
        'video': True,
        'audio': True,
        'memory': True,
        'posture': True,
        'behavior': True,
        'life': True,
        'touch': True,
        'speech': True,
        'sound': True   
    }
}

def load_config(config_path=None):
    """加载配置文件，如果不存在则使用默认配置"""
    # 首先使用默认配置
    config = DEFAULT_CONFIG.copy()
    
    # 尝试加载自定义配置
    try:
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
                
                # 合并配置
                if custom_config and isinstance(custom_config, dict):
                    # 更新robot部分
                    if 'robot' in custom_config:
                        config['robot'].update(custom_config['robot'])
                    
                    # 更新services部分
                    if 'services' in custom_config:
                        config['services'].update(custom_config['services'])
    except Exception as e:
        print("配置加载错误: %s" % str(e))
        
    return config

def save_config(config, config_path=None):
    """保存配置到文件"""
    if not config_path:
        config_dir = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')),
            "config"
        )
        config_path = os.path.join(config_dir, "robot_config.yaml")
    
    # 确保目录存在
    if not os.path.exists(os.path.dirname(config_path)):
        os.makedirs(os.path.dirname(config_path))
        
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return True
    except Exception as e:
        print("配置保存错误: %s" % str(e))
        return False