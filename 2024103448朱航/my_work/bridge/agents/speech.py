#!/usr/bin/env python
# -*- coding: utf-8 -*-

from naoqi import ALProxy
from . import BaseAgent
import traceback
import time

class SpeechAgent(BaseAgent):
    """Pepper机器人语音识别控制代理"""
    
    def __init__(self, ip, port):
        """初始化语音识别控制代理
        
        Args:
            ip (str): 机器人IP地址
            port (int): NAOqi端口号
        """
        super(SpeechAgent, self).__init__(ip, port)
        
    def _create_service_proxy(self):
        """创建语音识别服务代理"""
        self._proxies['ALSpeechRecognition'] = ALProxy("ALSpeechRecognition", self.ip, self.port)

    def register_methods(self):
        """注册供客户端调用的方法"""
        return {
            "speech": {
                "start_recognition": self.start_recognition,
                "stop_recognition": self.stop_recognition,
                "get_recognized_words": self.get_recognized_words,
                "get_available_languages": self.get_available_languages,
                "get_current_language": self.get_current_language,
                "set_vocabulary": self.set_vocabulary,
                "get_vocabulary": self.get_vocabulary,
                "set_parameter": self.set_parameter,
                "get_parameter": self.get_parameter,
                "is_recognition_active": self.is_recognition_active
            }
        }
    
    def get_speech_proxy(self):
        """获取语音识别代理"""
        return self.get_proxy("ALSpeechRecognition")
    
    def start_recognition(self, vocabulary=None, language=None):
        """启动语音识别
        
        Args:
            vocabulary (list): 词汇表，如果为None则使用默认词汇表
            language (str): 语言代码，如果为None则使用当前语言
            
        Returns:
            bool: 操作是否成功
        """
        try:
            asr_proxy = self.get_speech_proxy()
            
            # 停止之前的识别
            try:
                asr_proxy.unsubscribe("PepperBridge")
            except:
                pass
            
            # 设置词汇表
            if vocabulary is not None:
                asr_proxy.setVocabulary(vocabulary, False)
            
            # 设置语言
            if language is not None:
                asr_proxy.setLanguage(language)
            
            # 开始识别
            asr_proxy.subscribe("PepperBridge")
            
            if self._debug:
                print("开始语音识别")
            return True
        except Exception as e:
            if self._debug:
                print("启动语音识别失败: %s" % e)
                print(traceback.format_exc())
            return False
    
    def stop_recognition(self):
        """停止语音识别
        
        Returns:
            bool: 操作是否成功
        """
        try:
            asr_proxy = self.get_speech_proxy()
            asr_proxy.unsubscribe("PepperBridge")
            
            if self._debug:
                print("停止语音识别")
            return True
        except Exception as e:
            if self._debug:
                print("停止语音识别失败: %s" % e)
                print(traceback.format_exc())
            return False
    
    def get_recognized_words(self):
        """获取识别到的词语
        
        Returns:
            list: 识别到的词语列表，如果失败则返回空列表
        """
        try:
            asr_proxy = self.get_speech_proxy()
            words = asr_proxy.getRecognizedWords()
            
            if self._debug:
                print("识别到的词语: %s" % words)
            return words
        except Exception as e:
            if self._debug:
                print("获取识别词语失败: %s" % e)
                print(traceback.format_exc())
            return []
            
    def get_available_languages(self):
        """获取可用的语言列表
        
        Returns:
            list: 可用的语言代码列表
        """
        try:
            asr_proxy = self.get_speech_proxy()
            languages = asr_proxy.getAvailableLanguages()
            
            if self._debug:
                print("可用语言列表: %s" % languages)
            return languages
        except Exception as e:
            if self._debug:
                print("获取可用语言列表失败: %s" % e)
                print(traceback.format_exc())
            return []
            
    def get_current_language(self):
        """获取当前使用的语言
        
        Returns:
            str: 当前语言代码
        """
        try:
            asr_proxy = self.get_speech_proxy()
            language = asr_proxy.getLanguage()
            
            if self._debug:
                print("当前语言: %s" % language)
            return language
        except Exception as e:
            if self._debug:
                print("获取当前语言失败: %s" % e)
                print(traceback.format_exc())
            return None
            
    def set_vocabulary(self, vocabulary):
        """设置词汇表
        
        Args:
            vocabulary (list): 词汇表列表
            
        Returns:
            bool: 操作是否成功
        """
        try:
            asr_proxy = self.get_speech_proxy()
            asr_proxy.setVocabulary(vocabulary, False)
            
            if self._debug:
                print("设置词汇表: %s" % vocabulary)
            return True
        except Exception as e:
            if self._debug:
                print("设置词汇表失败: %s" % e)
                print(traceback.format_exc())
            return False
            
    def get_vocabulary(self):
        """获取当前词汇表
        
        Returns:
            list: 当前词汇表列表
        """
        try:
            asr_proxy = self.get_speech_proxy()
            vocabulary = asr_proxy.getVocabulary()
            
            if self._debug:
                print("当前词汇表: %s" % vocabulary)
            return vocabulary
        except Exception as e:
            if self._debug:
                print("获取词汇表失败: %s" % e)
                print(traceback.format_exc())
            return []
            
    def set_parameter(self, param, value):
        """设置识别参数
        
        Args:
            param (str): 参数名称
            value: 参数值
            
        Returns:
            bool: 操作是否成功
        """
        try:
            asr_proxy = self.get_speech_proxy()
            asr_proxy.setParameter(param, value)
            
            if self._debug:
                print("设置参数: %s = %s" % (param, value))
            return True
        except Exception as e:
            if self._debug:
                print("设置参数失败: %s" % e)
                print(traceback.format_exc())
            return False
            
    def get_parameter(self, param):
        """获取识别参数
        
        Args:
            param (str): 参数名称
            
        Returns:
            参数值，如果失败则返回None
        """
        try:
            asr_proxy = self.get_speech_proxy()
            value = asr_proxy.getParameter(param)
            
            if self._debug:
                print("获取参数: %s = %s" % (param, value))
            return value
        except Exception as e:
            if self._debug:
                print("获取参数失败: %s" % e)
                print(traceback.format_exc())
            return None
            
    def is_recognition_active(self):
        """检查语音识别是否处于活动状态
        
        Returns:
            bool: 是否正在识别
        """
        try:
            asr_proxy = self.get_speech_proxy()
            # 检查是否已订阅
            subscribers = asr_proxy.getSubscribersInfo()
            is_active = "PepperBridge" in [sub[0] for sub in subscribers]
            
            if self._debug:
                print("识别状态: %s" % ("活动" if is_active else "非活动"))
            return is_active
        except Exception as e:
            if self._debug:
                print("检查识别状态失败: %s" % e)
                print(traceback.format_exc())
            return False