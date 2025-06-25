#!/usr/bin/env python
# -*- coding: utf-8 -*-

from naoqi import ALProxy
from . import BaseAgent
import traceback
import time

class TTSAgent(BaseAgent):
    """Pepper机器人TTS控制代理"""
    
    def __init__(self, ip, port):
        """初始化TTS控制代理
        
        Args:
            ip (str): 机器人IP地址
            port (int): NAOqi端口号
        """
        super(TTSAgent, self).__init__(ip, port)
        
    def _create_service_proxy(self):
        """创建TTS服务代理"""
        self._proxies['ALTextToSpeech'] = ALProxy("ALTextToSpeech", self.ip, self.port)

    def register_methods(self):
        """注册供客户端调用的方法"""
        return {
            "tts": {
                "say": self.say,
                "set_language": self.set_language,
                "set_volume": self.set_volume,
                "set_parameter": self.set_parameter
            }
        }
    
    def get_tts_proxy(self):
        """获取TTS代理"""
        return self.get_proxy("ALTextToSpeech")
    
    def say(self, text):
        """让机器人说话
        
        Args:
            text (str): 要说的文本
            
        Returns:
            bool: 操作是否成功
        """
        try:
            # 确保文本是UTF-8编码的字符串
            if isinstance(text, unicode):
                text = text.encode('utf-8')
                
            if self._debug:
                print("让机器人说话: %s" % text)
                
            tts_proxy = self.get_tts_proxy()
            tts_proxy.say(text)
            return True
        except Exception as e:
            if self._debug:
                print("语音输出失败: %s" % e)
                print(traceback.format_exc())
            return False
    
    def set_language(self, language):
        """设置语音语言
        
        Args:
            language (str): 语言代码，如'Chinese'或'English'
            
        Returns:
            bool: 操作是否成功
        """
        try:
            # 确保语言代码正确
            valid_languages = {
                "chinese": "Chinese",
                "english": "English",
                "french": "French",
                "japanese": "Japanese",
                "korean": "Korean",
                "german": "German",
                "italian": "Italian",
                "spanish": "Spanish",
                "portuguese": "Portuguese",
                "brazilian": "Brazilian"
            }
            
            # 转换为标准格式
            language_lower = language.lower()
            if language_lower in valid_languages:
                language = valid_languages[language_lower]
            
            if self._debug:
                print("设置语言: %s" % language)
                
            tts_proxy = self.get_tts_proxy()
            tts_proxy.setLanguage(language)
            return True
        except Exception as e:
            if self._debug:
                print("设置语言异常: %s" % e)
                print(traceback.format_exc())
            return False
    
    def set_volume(self, volume):
        """设置语音音量
        
        Args:
            volume (float): 音量大小，范围0.0-1.0
            
        Returns:
            bool: 操作是否成功
        """
        try:
            # 确保音量在有效范围内
            volume = float(volume)
            if volume < 0.0:
                volume = 0.0
            elif volume > 1.0:
                volume = 1.0
                
            if self._debug:
                print("设置音量: %s" % volume)
                
            tts_proxy = self.get_tts_proxy()
            tts_proxy.setVolume(volume)
            return True
        except Exception as e:
            if self._debug:
                print("设置音量异常: %s" % e)
                print(traceback.format_exc())
            return False
    
    def set_parameter(self, param, value):
        """设置语音参数
        
        Args:
            param (str): 参数名称，如'speed'、'pitch'等
            value: 参数值
            
        Returns:
            bool: 操作是否成功
        """
        try:
            tts_proxy = self.get_tts_proxy()
            
            # 处理特殊参数
            if param.lower() == "speed":
                # 速度范围检查
                value = float(value)
                if value < 50:
                    value = 50
                elif value > 200:
                    value = 200
                    
                if self._debug:
                    print("设置语速: %s" % value)
                
                # 尝试不同的方法设置语速
                success = False
                errors = []
                
                # 方法1: 直接设置
                try:
                    tts_proxy.setParameter("speed", value)
                    success = True
                except Exception as e:
                    errors.append("方法1失败: %s" % e)
                    
                    # 方法2: 转换为整数
                    try:
                        tts_proxy.setParameter("speed", int(value))
                        success = True
                    except Exception as e:
                        errors.append("方法2失败: %s" % e)
                        
                        # 方法3: 转换为浮点数
                        try:
                            tts_proxy.setParameter("speed", float(value))
                            success = True
                        except Exception as e:
                            errors.append("方法3失败: %s" % e)
                            
                            # 方法4: 使用字符串
                            try:
                                tts_proxy.setParameter("speed", str(value))
                                success = True
                            except Exception as e:
                                errors.append("方法4失败: %s" % e)
                                
                                # 方法5: 使用另一个参数名
                                try:
                                    tts_proxy.setParameter("speechRate", value)
                                    success = True
                                except Exception as e:
                                    errors.append("方法5失败: %s" % e)
                
                if not success:
                    if self._debug:
                        print("所有设置语速方法都失败: %s" % errors)
                    return False
            else:
                # 其他参数
                if self._debug:
                    print("设置参数: %s = %s" % (param, value))
                tts_proxy.setParameter(param, value)
                
            return True
        except Exception as e:
            if self._debug:
                print("设置参数异常: %s" % e)
                print(traceback.format_exc())
            return False 