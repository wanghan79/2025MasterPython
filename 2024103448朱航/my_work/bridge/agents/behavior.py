#!/usr/bin/env python
# -*- coding: utf-8 -*-

from naoqi import ALProxy
from . import BaseAgent
import traceback
import time

class BehaviorAgent(BaseAgent):
    """Pepper机器人行为控制代理"""
    
    def __init__(self, ip, port):
        """初始化行为控制代理
        
        Args:
            ip (str): 机器人IP地址
            port (int): NAOqi端口号
        """
        super(BehaviorAgent, self).__init__(ip, port)
        
    def _create_service_proxy(self):
        """创建行为服务代理"""
        self._proxies['ALBehaviorManager'] = ALProxy("ALBehaviorManager", self.ip, self.port)
        self._proxies['ALRobotPosture'] = ALProxy("ALRobotPosture", self.ip, self.port)
        self._proxies['ALAutonomousLife'] = ALProxy("ALAutonomousLife", self.ip, self.port)
        self._proxies['ALMotion'] = ALProxy("ALMotion", self.ip, self.port)
        
    def _get_proxy(self, service_name):
        """获取指定服务的代理
        
        Args:
            service_name (str): 服务名称
            
        Returns:
            ALProxy: NAOqi服务代理
        """
        if service_name not in self._proxies:
            raise Exception("服务 %s 未初始化" % service_name)
        return self._proxies[service_name]
        
    def _get_behavior_proxy(self):
        """获取ALBehaviorManager代理"""
        return self._get_proxy("ALBehaviorManager")
    
    def _get_posture_proxy(self):
        """获取ALRobotPosture代理"""
        return self._get_proxy("ALRobotPosture")
    
    def _get_life_proxy(self):
        """获取ALAutonomousLife代理"""
        return self._get_proxy("ALAutonomousLife")

    def register_methods(self):
        """注册该代理类提供的远程调用方法"""
        return {
            'behavior': {
                'get_installed_behaviors': self.get_installed_behaviors,
                'play_animation': self._get_behavior_proxy().playAnimation,
                'stop_animation': self._get_behavior_proxy().stopAnimation,
                'run_behavior': self.run_behavior,
                'stop_behavior': self.stop_behavior,
                'stop_all_behaviors': self.stop_all_behaviors
            },
            'posture': {
                'go_to_posture': self.go_to_posture,
                'get_posture': self.get_posture
            },
            'life': {
                'set_autonomous_life_state': self.set_autonomous_life_state,
                'get_autonomous_life_state': self.get_autonomous_life_state
            }
        }
    
    def get_installed_behaviors(self):
        """获取已安装的行为列表
        
        Returns:
            list: 行为列表
        """
        try:
            behavior_proxy = self._get_behavior_proxy()
            
            if self._debug:
                print("获取已安装的行为列表")
            behaviors = behavior_proxy.getInstalledBehaviors()
            return behaviors
        except Exception as e:
            if self._debug:
                print("获取行为列表异常: %s" % e)
                print(traceback.format_exc())
            return []
    
    def run_behavior(self, behavior_name):
        """运行指定行为
        
        Args:
            behavior_name (str): 行为名称
            
        Returns:
            bool: 操作是否成功
        """
        try:
            behavior_proxy = self._get_behavior_proxy()
            
            # 检查行为是否存在
            behaviors = behavior_proxy.getInstalledBehaviors()
            if behavior_name not in behaviors:
                if self._debug:
                    print("行为 %s 不存在" % behavior_name)
                return False
            
            if self._debug:
                print("运行行为: %s" % behavior_name)
            
            # 尝试不同的方法运行行为
            success = False
            errors = []
            
            # 方法1: 直接运行
            try:
                result = behavior_proxy.runBehavior(behavior_name)
                if result:
                    success = True
                else:
                    errors.append("方法1失败: runBehavior返回False")
            except Exception as e:
                errors.append("方法1失败: %s" % e)
                
                # 方法2: 使用post异步运行
                try:
                    behavior_proxy.post.runBehavior(behavior_name)
                    success = True
                except Exception as e:
                    errors.append("方法2失败: %s" % e)
            
            if not success:
                if self._debug:
                    print("所有运行行为方法都失败: %s" % errors)
                return False
            
            return True
        except Exception as e:
            if self._debug:
                print("运行行为异常: %s" % e)
                print(traceback.format_exc())
            return False
    
    def stop_behavior(self, behavior_name):
        """停止指定行为
        
        Args:
            behavior_name (str): 行为名称
            
        Returns:
            bool: 操作是否成功
        """
        try:
            behavior_proxy = self._get_behavior_proxy()
            
            if self._debug:
                print("停止行为: %s" % behavior_name)
            behavior_proxy.stopBehavior(behavior_name)
            return True
        except Exception as e:
            if self._debug:
                print("停止行为异常: %s" % e)
                print(traceback.format_exc())
            return False
    
    def stop_all_behaviors(self):
        """停止所有行为
        
        Returns:
            bool: 操作是否成功
        """
        try:
            behavior_proxy = self._get_behavior_proxy()
            
            if self._debug:
                print("停止所有行为")
            behavior_proxy.stopAllBehaviors()
            return True
        except Exception as e:
            if self._debug:
                print("停止所有行为异常: %s" % e)
                print(traceback.format_exc())
            return False
    
    def go_to_posture(self, posture_name, speed):
        """设置机器人姿势
        
        Args:
            posture_name (str): 姿势名称，如'Stand'、'Crouch'等
            speed (float): 速度，范围0.0-1.0
            
        Returns:
            bool: 操作是否成功
        """
        try:
            # 确保速度在有效范围内
            speed = float(speed)
            if speed < 0.1:
                speed = 0.1
            elif speed > 1.0:
                speed = 1.0
            
            # 确保姿势名称正确（大小写敏感）
            valid_postures = ["Stand", "StandInit", "StandZero", "Crouch", "Sit", "SitRelax", "LyingBelly", "LyingBack"]
            
            # 如果姿势名称不在有效列表中，尝试查找最接近的
            if posture_name not in valid_postures:
                for valid_posture in valid_postures:
                    if valid_posture.lower() == posture_name.lower():
                        posture_name = valid_posture
                        break
            
            if self._debug:
                print("设置姿势: %s, 速度: %s" % (posture_name, speed))
            
            # 获取姿势代理
            posture_proxy = self._get_posture_proxy()
            
            # 尝试不同的方法设置姿势
            success = False
            errors = []
            
            # 方法1: 直接调用goToPosture
            try:
                result = posture_proxy.goToPosture(posture_name, speed)
                if result:
                    success = True
                else:
                    errors.append("方法1失败: goToPosture返回False")
            except Exception as e:
                errors.append("方法1失败: %s" % e)
                
                # 方法2: 使用motion代理
                try:
                    motion_proxy = self._get_proxy("ALMotion")
                        
                    if posture_name == "Stand" or posture_name == "StandInit" or posture_name == "StandZero":
                        motion_proxy.wakeUp()
                        success = True
                    elif posture_name == "Crouch":
                        motion_proxy.rest()
                        success = True
                    else:
                        errors.append("方法2失败: 不支持的姿势 %s" % posture_name)
                except Exception as e:
                    errors.append("方法2失败: %s" % e)
                    
                    # 方法3: 使用另一种形式的goToPosture
                    try:
                        result = posture_proxy.goToPosture(str(posture_name), float(speed))
                        if result:
                            success = True
                        else:
                            errors.append("方法3失败: goToPosture返回False")
                    except Exception as e:
                        errors.append("方法3失败: %s" % e)
            
            if not success:
                if self._debug:
                    print("所有设置姿势方法都失败: %s" % errors)
                return False
            
            return True
        except Exception as e:
            if self._debug:
                print("设置姿势异常: %s" % e)
                print(traceback.format_exc())
            return False
    
    def get_posture(self):
        """获取当前姿势
        
        Returns:
            str: 当前姿势名称
        """
        try:
            posture_proxy = self._get_posture_proxy()
            
            if self._debug:
                print("获取当前姿势")
            posture = posture_proxy.getPosture()
            return posture
        except Exception as e:
            if self._debug:
                print("获取姿势异常: %s" % e)
                print(traceback.format_exc())
            return "Unknown"
    
    def set_autonomous_life_state(self, state):
        """设置自主生命状态
        
        Args:
            state (str): 状态，可选值为'solitary'、'interactive'、'disabled'等
            
        Returns:
            bool: 操作是否成功
        """
        try:
            life_proxy = self._get_life_proxy()
            
            # 检查状态是否有效
            valid_states = ["solitary", "interactive", "safeguard", "disabled"]
            if state.lower() not in [s.lower() for s in valid_states]:
                if self._debug:
                    print("无效的自主生命状态: %s" % state)
                return False
            
            # 找到正确的状态名称（大小写敏感）
            for valid_state in valid_states:
                if valid_state.lower() == state.lower():
                    state = valid_state
                    break
            
            if self._debug:
                print("设置自主生命状态: %s" % state)
            life_proxy.setState(state)
            return True
        except Exception as e:
            if self._debug:
                print("设置自主生命状态异常: %s" % e)
                print(traceback.format_exc())
            return False
    
    def get_autonomous_life_state(self):
        """获取自主生命状态
        
        Returns:
            str: 当前状态
        """
        try:
            life_proxy = self._get_life_proxy()
            
            if self._debug:
                print("获取自主生命状态")
            state = life_proxy.getState()
            return state
        except Exception as e:
            if self._debug:
                print("获取自主生命状态异常: %s" % e)
                print(traceback.format_exc())
            return "Unknown"