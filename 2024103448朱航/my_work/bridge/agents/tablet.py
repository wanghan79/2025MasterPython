#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import BaseAgent
from naoqi import ALProxy
import traceback
import os
import time
from ..svg_face import PepperSVGLib, get_svg_service

class TabletAgent(BaseAgent):
    """Pepper机器人平板控制代理"""
    
    def __init__(self, ip, port):
        super(TabletAgent, self).__init__(ip, port)
        
    def _create_service_proxy(self):
        """创建平板服务代理"""
        try:
            if self._debug:
                print("尝试创建ALTabletService代理...")
                
            self._proxies['ALTabletService'] = ALProxy("ALTabletService", self.ip, self.port)
            
            if self._debug:
                print("ALTabletService代理创建成功")
                
            # 尝试重置平板
            try:
                self._proxies['ALTabletService'].hideWebview()
                time.sleep(0.5)
                self._proxies['ALTabletService'].resetTablet()
            except Exception as e:
                if self._debug:
                    print("重置平板失败: %s" % e)
                
        except Exception as e:
            if self._debug:
                print("创建ALTabletService代理失败: %s" % e)
                print(traceback.format_exc())
                
    def _get_tablet_proxy(self):
        """获取ALTabletService代理"""
        return self.get_proxy("ALTabletService")
    
    def show_image(self, image_path):
        """在平板上显示图片
        
        Args:
            image_path (str): 图片文件路径
            
        Returns:
            bool: 操作是否成功
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                if self._debug:
                    print("图片文件不存在: %s" % image_path)
                return False
                
            # 转换路径分隔符
            image_path = image_path.replace('\\', '/')
            
            # 获取平板代理
            tablet_proxy = self._get_tablet_proxy()
            if tablet_proxy is None:
                if self._debug:
                    print("无法获取平板代理")
                return False
                
            if self._debug:
                print("正在显示图片: %s" % image_path)
            
            # 先尝试隐藏当前网页
            try:
                tablet_proxy.hideWebview()
                time.sleep(0.5)
            except Exception as e:
                if self._debug:
                    print("隐藏网页失败: %s" % e)
            
            # 显示图片
            try:
                tablet_proxy.showImage(image_path)
                return True
            except Exception as e:
                if self._debug:
                    print("显示图片失败: %s" % e)
                    
                # 备选方案：通过网页显示图片
                try:
                    url = "file://%s" % image_path
                    tablet_proxy.showWebview(url)
                    return True
                except Exception as e2:
                    if self._debug:
                        print("通过网页显示图片也失败: %s" % e2)
                    return False
                
        except Exception as e:
            if self._debug:
                print("显示图片异常: %s" % e)
                print(traceback.format_exc())
            return False
    
    def show_web_page(self, url):
        """在平板上显示网页
        
        Args:
            url (str): 网页URL
            
        Returns:
            bool: 操作是否成功
        """
        try:
            tablet_proxy = self._get_tablet_proxy()
            tablet_proxy.showWebview(url)
            return True
        except Exception as e:
            if self._debug:
                print("显示网页异常: %s" % e)
                print(traceback.format_exc())
            return False
    
    def hide_web_page(self):
        """隐藏当前显示的网页
        
        Returns:
            bool: 操作是否成功
        """
        try:
            tablet_proxy = self._get_tablet_proxy()
            tablet_proxy.hideWebview()
            return True
        except Exception as e:
            if self._debug:
                print("隐藏网页异常: %s" % e)
            return False
    
    def execute_js(self, js_code):
        """在平板当前网页中执行JavaScript代码
        
        Args:
            js_code (str): JavaScript代码
            
        Returns:
            bool: 操作是否成功
        """
        try:
            tablet_proxy = self._get_tablet_proxy()
            tablet_proxy.executeJS(js_code)
            return True
        except Exception as e:
            if self._debug:
                print("执行JS异常: %s" % e)
            return False
    
    def reset_tablet(self):
        """重置平板显示
        
        Returns:
            bool: 操作是否成功
        """
        try:
            tablet_proxy = self._get_tablet_proxy()
            tablet_proxy.resetTablet()
            return True
        except Exception as e:
            if self._debug:
                print("重置平板异常: %s" % e)
            return False
    
    def show_emoji(self, emoji_name, title=None, say_name=False):
        """在平板上显示SVG表情
        
        Args:
            emoji_name (str): 表情名称，如'smile', 'sad', 'surprise', 'love', 'think'
            title (str, optional): 表情的标题文字，默认为None（使用默认标题）
            say_name (bool, optional): 是否同时说出表情名称，默认为False
            
        Returns:
            bool: 操作是否成功
        """
        try:
            if self._debug:
                print(u"正在显示SVG表情: %s" % emoji_name)
                
            # 创建SVG服务
            svg_service = get_svg_service(self.ip, self.port, self._debug)
            
            # 检查是否需要连接
            if not svg_service.connected:
                svg_service.connect()
                
            # 显示表情
            result = svg_service.show_emoji(emoji_name, title, say_name)
            
            return result
        except Exception as e:
            if self._debug:
                print(u"显示SVG表情失败: %s" % e)
                print(traceback.format_exc())
            return False
            
    def get_available_emojis(self):
        """获取可用的SVG表情列表
        
        Returns:
            list: 可用表情名称列表
        """
        try:
            svg_service = get_svg_service(self.ip, self.port, self._debug)
            return svg_service.get_available_emojis()
        except Exception as e:
            if self._debug:
                print(u"获取可用表情列表失败: %s" % e)
            return []
    
    def register_methods(self):
        """注册代理方法"""
        return {
            "tablet": {
                "show_image": self.show_image,
                "show_web_page": self.show_web_page,
                "hide_web_page": self.hide_web_page,
                "execute_js": self.execute_js,
                "reset_tablet": self.reset_tablet,
                "show_emoji": self.show_emoji,
                "get_available_emojis": self.get_available_emojis
            }
        }