#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import BaseAgent
from naoqi import ALProxy
import traceback
import time
import math
import inspect

class MotionAgent(BaseAgent):
    """Pepper机器人运动控制代理"""
    
    def __init__(self, ip, port):
        super(MotionAgent, self).__init__(ip, port)
        
    def _create_service_proxy(self):
        """创建运动服务代理"""
        self._proxies['ALMotion'] = ALProxy("ALMotion", self.ip, self.port)
        self._proxies['ALAutonomousLife'] = ALProxy("ALAutonomousLife", self.ip, self.port)
        
    def _get_motion_proxy(self):
        """获取ALMotion代理"""
        return self.get_proxy("ALMotion")
    
    def restore_autonomous_life(self):
        """恢复机器人的自主生命状态
        
        Returns:
            bool: 操作是否成功
        """
        try:
            # 获取自主生命代理
            life_proxy = self.get_proxy("ALAutonomousLife")
            
            if self._debug:
                print("正在恢复自主生命状态...")
            
            # 设置状态为solitary（独立模式）
            life_proxy.setState("solitary")
            
            # 等待状态变化
            time.sleep(2)
            
            # 检查状态
            current_state = life_proxy.getState()
            if self._debug:
                print("当前自主生命状态: %s" % current_state)
            
            if current_state != "solitary":
                if self._debug:
                    print("无法恢复自主生命状态，当前状态: %s" % current_state)
                return False
            
            return True
        except Exception as e:
            if self._debug:
                print("恢复自主生命状态异常: %s" % e)
                print(traceback.format_exc())
            return False
    
    def ensure_robot_ready(self):
        """确保机器人已准备好进行运动控制
        
        Returns:
            bool: 机器人是否准备好
        """
        try:
            if self._debug:
                print("确保机器人准备就绪...")
            
            motion_proxy = self._get_motion_proxy()
            
            # 1. 禁用自主生命模式（这是关键）
            try:
                life_proxy = self.get_proxy("ALAutonomousLife")
                current_state = life_proxy.getState()
                
                if self._debug:
                    print("当前自主生命状态: %s" % current_state)
                
                if current_state != "disabled":
                    if self._debug:
                        print("正在禁用自主生命...")
                    
                    # 先关闭当前活动
                    if current_state == "interactive" or current_state == "solitary":
                        try:
                            life_proxy.stopActivity()
                        except:
                            if self._debug:
                                print("停止活动失败，继续进行")
                    
                    # 禁用自主生命
                    life_proxy.setState("disabled")
                    time.sleep(1)  # 等待状态变化
            except Exception as e:
                if self._debug:
                    print("禁用自主生命失败: %s" % e)
            
            # 2. 确保机器人醒着
            if hasattr(motion_proxy, "wakeUp"):
                motion_proxy.wakeUp()
            
            # 3. 停止任何可能的移动
            if hasattr(motion_proxy, "stopMove"):
                motion_proxy.stopMove()
            
            # 4. 设置全身刚度为1.0以启用运动控制
            motion_proxy.setStiffnesses("Body", 1.0)
            
            if self._debug:
                print("机器人已准备就绪")
                
            return True
            
        except Exception as e:
            if self._debug:
                print("准备机器人失败: %s" % e)
                print(traceback.format_exc())
            return False
    
    def move_to(self, x, y, theta):
        """移动机器人到指定位置
        
        Args:
            x (float): X轴移动距离（米）
            y (float): Y轴移动距离（米）
            theta (float): 旋转角度（弧度）
            
        Returns:
            bool: 操作是否成功
        """
        try:
            # 确保参数类型正确
            x = float(x)
            y = float(y)
            theta = float(theta)
            
            if self._debug:
                print("移动到: x=%s, y=%s, theta=%s" % (x, y, theta))
            
            # 获取运动代理
            motion_proxy = self._get_motion_proxy()
            
            # 直接使用NAOqi原生的moveTo方法
            motion_proxy.moveTo(x, y, theta)
            
            if self._debug:
                print("移动完成")
            
            return True
            
        except Exception as e:
            if self._debug:
                print("移动异常: %s" % e)
                print(traceback.format_exc())
            return False
    
    def stop_move(self):
        """停止所有运动
        
        Returns:
            bool: 操作是否成功
        """
        try:
            motion_proxy = self._get_motion_proxy()
            motion_proxy.stopMove()
            return True
        except Exception as e:
            if self._debug:
                print("停止运动异常: %s" % e)
            return False
    
    def move(self, x, y, theta):
        """设置机器人运动速度
        
        Args:
            x (float): 前进/后退速度（-1.0到1.0）
            y (float): 左/右速度（-1.0到1.0）
            theta (float): 旋转速度（-1.0到1.0）
            
        Returns:
            bool: 操作是否成功
        """
        try:
            # 确保机器人已准备好
            self.ensure_robot_ready()
            
            motion_proxy = self._get_motion_proxy()
            if self._debug:
                print("设置速度: x=%s, y=%s, theta=%s" % (x, y, theta))
            motion_proxy.move(x, y, theta)
            return True
        except Exception as e:
            if self._debug:
                print("设置速度异常: %s" % e)
            return False
    
    def rest(self):
        """让机器人进入休息状态
        
        Returns:
            bool: 操作是否成功
        """
        try:
            motion_proxy = self._get_motion_proxy()
            if self._debug:
                print("进入休息状态")
            motion_proxy.rest()
            return True
        except Exception as e:
            if self._debug:
                print("进入休息状态异常: %s" % e)
            return False
    
    def wake_up(self):
        """唤醒机器人
        
        Returns:
            bool: 操作是否成功
        """
        try:
            motion_proxy = self._get_motion_proxy()
            if self._debug:
                print("唤醒机器人")
            motion_proxy.wakeUp()
            return True
        except Exception as e:
            if self._debug:
                print("唤醒机器人异常: %s" % e)
            return False
    
    def turn(self, angle):
        """控制机器人原地旋转
        
        Args:
            angle (float): 旋转角度（弧度）
            
        Returns:
            bool: 操作是否成功
        """
        try:
            # 确保参数类型正确
            angle = float(angle)
            
            if self._debug:
                print("旋转角度: %s弧度" % angle)
            
            # 确保机器人已准备好
            if not self.ensure_robot_ready():
                if self._debug:
                    print("机器人未准备好，无法旋转")
                return False
            
            # 获取运动代理
            motion_proxy = self._get_motion_proxy()
            
            # 尝试不同的旋转方法
            success = False
            errors = []
            
            # 方法1: 使用moveTo旋转
            try:
                if self._debug:
                    print("尝试方法1: moveTo(0, 0, %s)" % angle)
                motion_proxy.moveTo(0, 0, angle)
                if self._debug:
                    print("方法1成功")
                success = True
            except Exception as e:
                errors.append("方法1失败: %s" % e)
                if self._debug:
                    print("方法1失败: %s" % e)
                
                # 方法2: 使用moveToward旋转
                try:
                    if self._debug:
                        print("尝试方法2: moveToward + stopMove")
                    # 先启动旋转
                    motion_proxy.moveToward(0, 0, angle/abs(angle) * 0.2)  # 使用较小的角速度
                    if self._debug:
                        print("moveToward已启动")
                    # 等待一段时间
                    time.sleep(max(1, abs(angle)))
                    # 停止移动
                    motion_proxy.stopMove()
                    if self._debug:
                        print("方法2成功")
                    success = True
                except Exception as e:
                    errors.append("方法2失败: %s" % e)
                    if self._debug:
                        print("方法2失败: %s" % e)
                    
                    # 方法3: 使用异步moveTo
                    try:
                        if self._debug:
                            print("尝试方法3: 异步moveTo")
                        motion_proxy.post.moveTo(0, 0, angle)
                        if self._debug:
                            print("方法3成功")
                        success = True
                    except Exception as e:
                        errors.append("方法3失败: %s" % e)
                        if self._debug:
                            print("方法3失败: %s" % e)
                        
                        # 方法4: 使用angleInterpolation
                        try:
                            if self._debug:
                                print("尝试方法4: angleInterpolation")
                            names = ["HeadYaw"]
                            angles = [angle]
                            times = [2.0]
                            motion_proxy.angleInterpolation(names, angles, times, True)
                            if self._debug:
                                print("方法4成功")
                            success = True
                        except Exception as e:
                            errors.append("方法4失败: %s" % e)
                            if self._debug:
                                print("方法4失败: %s" % e)
            
            if not success:
                if self._debug:
                    print("所有旋转方法都失败: %s" % errors)
                return False
            
            return True
        except Exception as e:
            if self._debug:
                print("旋转异常: %s" % e)
                print(traceback.format_exc())
            return False
    
    def set_velocity(self, x, y, theta):
        """设置机器人的速度
        
        Args:
            x (float): 前进/后退速度（米/秒）
            y (float): 左/右速度（米/秒）
            theta (float): 旋转速度（弧度/秒）
            
        Returns:
            bool: 操作是否成功
        """
        try:
            # 确保参数类型正确
            x = float(x)
            y = float(y)
            theta = float(theta)
            
            if self._debug:
                print("设置速度: x=%s, y=%s, theta=%s" % (x, y, theta))
            
            # 强制限制参数范围（-1.0到1.0）
            x = max(-1.0, min(1.0, x))
            y = max(-1.0, min(1.0, y))
            theta = max(-1.0, min(1.0, theta))
            
            # 获取运动代理
            motion_proxy = self._get_motion_proxy()
            
            # 直接使用NAOqi原生的moveToward方法
            motion_proxy.moveToward(x, y, theta)
            
            if self._debug:
                print("设置速度成功")
            
            return True
                
        except Exception as e:
            if self._debug:
                print("设置速度异常: %s" % e)
                print(traceback.format_exc())
            return False
    
    def set_arm_angles(self, arm, angles, speed=0.1):
        """设置手臂关节角度
        
        Args:
            arm (str): 'L' 或 'R'，表示左臂或右臂
            angles (list): 关节角度列表，单位是弧度
            speed (float): 运动速度，范围0.0-1.0
            
        Returns:
            bool: 操作是否成功
        """
        try:
            # 确保机器人已准备好
            if not self.ensure_robot_ready():
                if self._debug:
                    print("机器人未准备好")
                return False
                
            # 设置手臂关节名称
            if arm.upper() == 'L':
                joint_names = [
                    'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw'
                ]
            elif arm.upper() == 'R':
                joint_names = [
                    'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw'
                ]
            else:
                if self._debug:
                    print("无效的手臂参数，请使用 'L' 或 'R'")
                return False
                
            # 确保角度列表长度正确
            if len(angles) != len(joint_names):
                if self._debug:
                    print("角度列表长度错误，需要 %d 个角度值" % len(joint_names))
                return False
                
            # 获取运动代理
            motion_proxy = self._get_motion_proxy()
            
            # 设置关节角度
            motion_proxy.setAngles(joint_names, angles, speed)
            
            if self._debug:
                print("%s臂已移动到指定位置" % arm)
            
            return True
            
        except Exception as e:
            if self._debug:
                print("设置手臂角度异常: %s" % e)
                print(traceback.format_exc())
            return False
            
    def set_hand_open(self, hand, open_percentage, speed=0.1):
        """设置手指开合程度
        
        Args:
            hand (str): 'L' 或 'R'，表示左手或右手
            open_percentage (float): 开合程度，范围0.0-1.0
            speed (float): 运动速度，范围0.0-1.0
            
        Returns:
            bool: 操作是否成功
        """
        try:
            # 确保机器人已准备好
            if not self.ensure_robot_ready():
                if self._debug:
                    print("机器人未准备好")
                return False
                
            # 设置手部关节名称
            if hand.upper() == 'L':
                joint_name = 'LHand'
            elif hand.upper() == 'R':
                joint_name = 'RHand'
            else:
                if self._debug:
                    print("无效的手部参数，请使用 'L' 或 'R'")
                return False
                
            # 确保开合程度在有效范围内
            open_percentage = max(0.0, min(1.0, open_percentage))
            
            # 获取运动代理
            motion_proxy = self._get_motion_proxy()
            
            # 设置手指开合
            motion_proxy.setAngles([joint_name], [open_percentage], speed)
            
            if self._debug:
                print("%s手已调整到 %.1f%% 开合度" % (hand, open_percentage * 100))
            
            return True
            
        except Exception as e:
            if self._debug:
                print("设置手指开合异常: %s" % e)
                print(traceback.format_exc())
            return False
            
    def wave_hand(self, hand='R', times=3):
        """执行挥手动作
        
        Args:
            hand (str): 'L' 或 'R'，表示左手或右手
            times (int): 挥手次数
            
        Returns:
            bool: 操作是否成功
        """
        try:
            # 确保机器人已准备好
            if not self.ensure_robot_ready():
                if self._debug:
                    print("机器人未准备好")
                return False
                
            # 设置手臂关节名称
            if hand.upper() == 'L':
                shoulder_pitch = 'LShoulderPitch'
                shoulder_roll = 'LShoulderRoll'
            elif hand.upper() == 'R':
                shoulder_pitch = 'RShoulderPitch'
                shoulder_roll = 'RShoulderRoll'
            else:
                if self._debug:
                    print("无效的手部参数，请使用 'L' 或 'R'")
                return False
                
            # 获取运动代理
            motion_proxy = self._get_motion_proxy()
            
            # 挥手动作
            for _ in range(times):
                # 抬起手臂
                motion_proxy.setAngles([shoulder_pitch], [0.0], 0.1)
                time.sleep(0.5)
                
                # 左右摆动
                motion_proxy.setAngles([shoulder_roll], [0.5], 0.1)
                time.sleep(0.3)
                motion_proxy.setAngles([shoulder_roll], [-0.5], 0.1)
                time.sleep(0.3)
                
            # 放下手臂
            motion_proxy.setAngles([shoulder_pitch], [1.5], 0.1)
            
            if self._debug:
                print("%s手已完成 %d 次挥手" % (hand, times))
            
            return True
            
        except Exception as e:
            if self._debug:
                print("挥手动作异常: %s" % e)
                print(traceback.format_exc())
            return False
    
    def transformPosition(self, source_frame, target_frame, position):
        """将位置从一个坐标系转换到另一个坐标系
        
        Args:
            source_frame (str): 源坐标系名称，如 'CameraTop' 或 'CameraBottom'
            target_frame (str): 目标坐标系名称，如 'Torso'
            position (list): 要转换的位置 [x, y, z]
            
        Returns:
            list: 转换后的位置 [x, y, z]
        """
        try:
            # 确保机器人已准备好
            if not self.ensure_robot_ready():
                if self._debug:
                    print("机器人未准备好")
                return None
                
            # 获取运动代理
            motion_proxy = self._get_motion_proxy()
            
            # 使用ALMotion的transformPosition方法进行坐标转换
            transformed_position = motion_proxy.transformPosition(source_frame, target_frame, position)
            
            if self._debug:
                print("坐标转换: %s -> %s" % (source_frame, target_frame))
                print("原始位置: %s" % position)
                print("转换后位置: %s" % transformed_position)
            
            return transformed_position
            
        except Exception as e:
            if self._debug:
                print("坐标转换异常: %s" % e)
                print(traceback.format_exc())
            return None
    
    def setPositions(self, effector, frame, position, fractionMaxSpeed=0.2, axisMask=7):
        """将末端执行器移动到指定位置
        
        Args:
            effector (str): 末端执行器名称，如 'RArm' 或 'LArm'
            frame (int): 参考坐标系，如 motion.FRAME_TORSO
            position (list): 目标位置 [x, y, z, wx, wy, wz]，单位为米和弧度
            fractionMaxSpeed (float): 最大速度比例，范围0.0-1.0
            axisMask (int): 控制轴掩码，7表示控制x,y,z三个方向
            
        Returns:
            bool: 操作是否成功
        """
        try:
            # 确保机器人已准备好
            if not self.ensure_robot_ready():
                if self._debug:
                    print("机器人未准备好")
                return False
                
            # 获取运动代理
            motion_proxy = self._get_motion_proxy()
            
            if self._debug:
                print("移动 %s 到位置: %s" % (effector, position))
                print("参考坐标系: %s" % frame)
                print("速度比例: %s" % fractionMaxSpeed)
                print("控制轴: %s" % axisMask)
            
            # 调用ALMotion的setPositions方法
            motion_proxy.setPositions(
                effector,
                frame,
                position,
                fractionMaxSpeed,
                axisMask
            )
            
            if self._debug:
                print("%s 移动完成" % effector)
            
            return True
            
        except Exception as e:
            if self._debug:
                print("移动 %s 异常: %s" % (effector, e))
                print(traceback.format_exc())
            return False
    
    def move_forward(self, distance, speed=0.3):
        """向前移动指定距离
        
        Args:
            distance (float): 移动距离（米）
            speed (float): 移动速度，范围0.0-1.0
            
        Returns:
            bool: 操作是否成功
        """
        try:
            if self._debug:
                print("向前移动 %.2f 米" % distance)
            
            motion_proxy = self._get_motion_proxy()
            motion_proxy.moveTo(distance, 0, 0, speed)
            return True
        except Exception as e:
            if self._debug:
                print("向前移动失败: %s" % e)
            return False
            
    def move_backward(self, distance, speed=0.3):
        """向后移动指定距离
        
        Args:
            distance (float): 移动距离（米）
            speed (float): 移动速度，范围0.0-1.0
            
        Returns:
            bool: 操作是否成功
        """
        try:
            if self._debug:
                print("向后移动 %.2f 米" % distance)
            
            motion_proxy = self._get_motion_proxy()
            motion_proxy.moveTo(-distance, 0, 0, speed)
            return True
        except Exception as e:
            if self._debug:
                print("向后移动失败: %s" % e)
            return False
            
    def move_left(self, distance, speed=0.3):
        """向左移动指定距离
        
        Args:
            distance (float): 移动距离（米）
            speed (float): 移动速度，范围0.0-1.0
            
        Returns:
            bool: 操作是否成功
        """
        try:
            if self._debug:
                print("向左移动 %.2f 米" % distance)
            
            motion_proxy = self._get_motion_proxy()
            motion_proxy.moveTo(0, distance, 0, speed)
            return True
        except Exception as e:
            if self._debug:
                print("向左移动失败: %s" % e)
            return False
            
    def move_right(self, distance, speed=0.3):
        """向右移动指定距离
        
        Args:
            distance (float): 移动距离（米）
            speed (float): 移动速度，范围0.0-1.0
            
        Returns:
            bool: 操作是否成功
        """
        try:
            if self._debug:
                print("向右移动 %.2f 米" % distance)
            
            motion_proxy = self._get_motion_proxy()
            motion_proxy.moveTo(0, -distance, 0, speed)
            return True
        except Exception as e:
            if self._debug:
                print("向右移动失败: %s" % e)
            return False
    
    def register_methods(self):
        """注册该代理类提供的远程调用方法"""
        return {
            'motion': {
                'moveTo': self.move_to,
                'stopMove': self.stop_move,
                'move': self.set_velocity,
                'setVelocity': self.set_velocity,
                'rest': self.rest,
                'wakeUp': self.wake_up,
                'turn': self.turn,
                'setArmAngles': self.set_arm_angles,
                'setHandOpen': self.set_hand_open,
                'waveHand': self.wave_hand,
                'transformPosition': self.transformPosition,
                'setPositions': self.setPositions,
                'moveForward': self.move_forward,
                'moveBackward': self.move_backward,
                'moveLeft': self.move_left,
                'moveRight': self.move_right
            }
        }