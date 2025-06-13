#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import traceback
import numpy as np

from . import BaseAgent
from naoqi import ALProxy

class MathAgent(BaseAgent):
    """Pepper机器人数学变换和运动学计算代理"""
    
    def __init__(self, ip, port):
        """初始化数学变换代理
        
        Args:
            ip (str): 机器人IP地址
            port (int): NAOqi端口号
        """
        super(MathAgent, self).__init__(ip, port)
        self._debug = True
        
    def _create_service_proxy(self):
        """创建服务代理"""
        try:
            if self._debug:
                print("创建ALMath代理...")
            self._proxies['ALMath'] = ALProxy("ALMath", self.ip, self.port)
            if self._debug:
                print("代理创建成功")
        except Exception as e:
            if self._debug:
                print("创建代理失败: %s" % e)
                print(traceback.format_exc())
            
    def create_transform(self, x=0.0, y=0.0, z=0.0, rx=0.0, ry=0.0, rz=0.0):
        """创建4x4变换矩阵
        
        Args:
            x (float): X轴平移
            y (float): Y轴平移
            z (float): Z轴平移
            rx (float): X轴旋转角度（弧度）
            ry (float): Y轴旋转角度（弧度）
            rz (float): Z轴旋转角度（弧度）
            
        Returns:
            list: 4x4变换矩阵的列表表示
        """
        try:
            math_proxy = self.get_proxy("ALMath")
            if math_proxy is None:
                return None
                
            # 使用ALMath服务创建变换矩阵
            transform = math_proxy.createTransform(x, y, z, rx, ry, rz)
            return transform
        except Exception as e:
            if self._debug:
                print("创建变换矩阵失败: %s" % e)
                print(traceback.format_exc())
            return None
        
    def transform_point(self, transform, point):
        """将点从一个坐标系转换到另一个
        
        Args:
            transform (list): 4x4变换矩阵的列表表示
            point (list): 要转换的点的坐标 [x, y, z]
            
        Returns:
            list: 转换后的点坐标 [x, y, z]
        """
        try:
            math_proxy = self.get_proxy("ALMath")
            if math_proxy is None:
                return None
                
            # 使用ALMath服务进行坐标转换
            transformed_point = math_proxy.transformPoint(transform, point)
            return transformed_point
        except Exception as e:
            if self._debug:
                print("坐标转换失败: %s" % e)
                print(traceback.format_exc())
            return None
        
    def get_robot_pose(self):
        """获取机器人当前位姿
        
        Returns:
            dict: 包含位置和角度的字典
        """
        try:
            motion = self.get_proxy("ALMotion")
            if motion is None:
                return None
                
            # 获取机器人位置和角度
            position = motion.getRobotPosition(False)  # False表示不使用传感器融合
            angles = motion.getAngles("Body", False)
            
            return {
                'position': {
                    'x': position[0],
                    'y': position[1],
                    'z': position[2]
                },
                'angles': {
                    'rx': angles[0],
                    'ry': angles[1],
                    'rz': angles[2]
                }
            }
        except Exception as e:
            if self._debug:
                print("获取机器人位姿失败: %s" % e)
                print(traceback.format_exc())
            return None
            
    def interpolate_pose(self, start_pose, end_pose, steps=10):
        """在两个位姿之间进行插值
        
        Args:
            start_pose (dict): 起始位姿
            end_pose (dict): 目标位姿
            steps (int): 插值步数
            
        Returns:
            list: 插值后的位姿序列
        """
        try:
            math_proxy = self.get_proxy("ALMath")
            if math_proxy is None:
                return None
                
            # 准备起始和目标位姿数据
            start_data = [
                start_pose['position']['x'],
                start_pose['position']['y'],
                start_pose['position']['z'],
                start_pose['angles']['rx'],
                start_pose['angles']['ry'],
                start_pose['angles']['rz']
            ]
            
            end_data = [
                end_pose['position']['x'],
                end_pose['position']['y'],
                end_pose['position']['z'],
                end_pose['angles']['rx'],
                end_pose['angles']['ry'],
                end_pose['angles']['rz']
            ]
            
            # 使用ALMath服务进行插值
            poses = math_proxy.interpolatePose(start_data, end_data, steps)
            
            # 格式化返回数据
            formatted_poses = []
            for pose in poses:
                formatted_poses.append({
                    'position': {
                        'x': pose[0],
                        'y': pose[1],
                        'z': pose[2]
                    },
                    'angles': {
                        'rx': pose[3],
                        'ry': pose[4],
                        'rz': pose[5]
                    }
                })
                
            return formatted_poses
        except Exception as e:
            if self._debug:
                print("位姿插值失败: %s" % e)
                print(traceback.format_exc())
            return None
        
    def register_methods(self):
        """注册代理方法
        
        Returns:
            dict: 方法映射字典，格式为 {'service_name': {'method_name': method}}
        """
        return {
            'math': {
                'create_transform': self.create_transform,
                'transform_point': self.transform_point,
                'get_robot_pose': self.get_robot_pose,
                'interpolate_pose': self.interpolate_pose
            }
        }