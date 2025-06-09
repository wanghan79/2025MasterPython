#!/usr/bin/env python
# -*- coding: utf-8 -*-

from naoqi import ALProxy
from . import BaseAgent
import numpy as np
import time
import traceback

class VisionAgent(BaseAgent):
    """Pepper机器人视觉控制代理"""
    
    def __init__(self, ip, port):
        """初始化视觉控制代理
        
        Args:
            ip (str): 机器人IP地址
            port (int): NAOqi端口号
        """
        super(VisionAgent, self).__init__(ip, port)
        self._camera_subscription = None
        
    def _create_service_proxy(self):
        """创建视觉服务代理"""
        self._proxies['ALVideoDevice'] = ALProxy("ALVideoDevice", self.ip, self.port)
        self._proxies['ALMemory'] = ALProxy("ALMemory", self.ip, self.port)
        self._proxies['ALFaceDetection'] = ALProxy("ALFaceDetection", self.ip, self.port)
        
    def _get_video_proxy(self):
        """获取ALVideoDevice代理"""
        return self.get_proxy("ALVideoDevice")
    
    def _get_memory_proxy(self):
        """获取ALMemory代理"""
        return self.get_proxy("ALMemory")
    
    def take_picture(self, camera_id=0, resolution=2, color_space=11):
        """拍照
        
        Args:
            camera_id (int): 相机ID，0为顶部相机，1为底部相机，2为深度相机
            resolution (int): 分辨率，0=QQVGA, 1=QVGA, 2=VGA, 3=4VGA
            color_space (int): 颜色空间，0=Y通道, 9=YUV, 10=RGB, 11=BGR
            
        Returns:
            numpy.ndarray: 图像数据，失败则返回None
        """
        try:
            proxy = self._get_video_proxy()
            
            # 订阅相机
            client_name = "python_client_%d" % int(time.time())
            if self._debug:
                print("拍照: 相机ID=%s, 分辨率=%s, 颜色空间=%s" % (camera_id, resolution, color_space))
                
            self._camera_subscription = proxy.subscribeCamera(
                client_name, camera_id, resolution, color_space, 30
            )
            
            # 获取图像
            image = proxy.getImageRemote(self._camera_subscription)
            
            # 取消订阅
            proxy.unsubscribe(self._camera_subscription)
            self._camera_subscription = None
            
            if image is None:
                if self._debug:
                    print("获取图像失败: 返回为空")
                return None
                
            # 解析图像数据
            width, height = image[0], image[1]
            image_data = np.frombuffer(image[6], dtype=np.uint8)
            
            if color_space == 0:  # Y通道
                return image_data.reshape((height, width))
            else:  # YUV, RGB, BGR
                channels = 3
                return image_data.reshape((height, width, channels))
                
        except Exception as e:
            if self._debug:
                print("拍照失败: %s" % e)
                print(traceback.format_exc())
            if self._camera_subscription:
                try:
                    self._get_video_proxy().unsubscribe(self._camera_subscription)
                    self._camera_subscription = None
                except:
                    pass
            return None
    
    def start_video_stream(self, camera_id=0, resolution=1, color_space=11):
        """开始视频流
        
        Args:
            camera_id (int): 相机ID，0为顶部相机，1为底部相机，2为深度相机
            resolution (int): 分辨率，0=QQVGA, 1=QVGA, 2=VGA, 3=4VGA
            color_space (int): 颜色空间，0=Y通道, 9=YUV, 10=RGB, 11=BGR
            
        Returns:
            str: 视频流订阅ID，失败则返回None
        """
        try:
            proxy = self._get_video_proxy()
            
            # 订阅相机
            client_name = "python_stream_%d" % int(time.time())
            if self._debug:
                print("开始视频流: 相机ID=%s, 分辨率=%s, 颜色空间=%s" % (camera_id, resolution, color_space))
                
            subscription_id = proxy.subscribeCamera(
                client_name, camera_id, resolution, color_space, 30
            )
            
            return subscription_id
                
        except Exception as e:
            if self._debug:
                print("开始视频流失败: %s" % e)
                print(traceback.format_exc())
            return None
    
    def get_video_frame(self, subscription_id):
        """获取视频帧
        
        Args:
            subscription_id (str): 视频流订阅ID
            
        Returns:
            numpy.ndarray: 图像数据，失败则返回None
        """
        try:
            proxy = self._get_video_proxy()
            
            # 获取图像
            if self._debug:
                print("获取视频帧: 订阅ID=%s" % subscription_id)
                
            image = proxy.getImageRemote(subscription_id)
            
            if image is None:
                if self._debug:
                    print("获取视频帧失败: 返回为空")
                return None
                
            # 解析图像数据
            width, height = image[0], image[1]
            color_space = image[2]
            image_data = np.frombuffer(image[6], dtype=np.uint8)
            
            if color_space == 0:  # Y通道
                return image_data.reshape((height, width))
            else:  # YUV, RGB, BGR
                channels = 3
                return image_data.reshape((height, width, channels))
                
        except Exception as e:
            if self._debug:
                print("获取视频帧失败: %s" % e)
                print(traceback.format_exc())
            return None
    
    def stop_video_stream(self, subscription_id):
        """停止视频流
        
        Args:
            subscription_id (str): 视频流订阅ID
            
        Returns:
            bool: 操作是否成功
        """
        try:
            proxy = self._get_video_proxy()
            if self._debug:
                print("停止视频流: 订阅ID=%s" % subscription_id)
                
            proxy.unsubscribe(subscription_id)
            return True
                
        except Exception as e:
            if self._debug:
                print("停止视频流失败: %s" % e)
                print(traceback.format_exc())
            return False
    
    def detect_faces(self):
        """检测人脸
        
        Returns:
            list: 检测到的人脸信息
        """
        try:
            # 获取人脸检测代理
            face_proxy = self.get_proxy("ALFaceDetection")
            # 获取内存代理
            memory_proxy = self._get_memory_proxy()
            
            # 订阅人脸检测事件
            face_proxy.subscribe("PepperFaceDetection")
            
            # 等待一段时间让代理检测到人脸
            time.sleep(1.0)
            
            # 从内存中获取人脸信息
            face_data = memory_proxy.getData("FaceDetected")
            
            # 取消订阅
            face_proxy.unsubscribe("PepperFaceDetection")
            
            # 解析人脸数据
            if face_data and len(face_data) > 1:
                # 有人脸被检测到
                faces = []
                face_info_array = face_data[1]
                
                for i in range(len(face_info_array) - 1):
                    face_info = face_info_array[i]
                    face = {
                        "shape_info": face_info[0],
                        "extra_info": face_info[1]
                    }
                    faces.append(face)
                
                return faces
            else:
                # 没有检测到人脸
                return []
        
        except Exception as e:
            if self._debug:
                print("检测人脸异常: %s" % e)
                print(traceback.format_exc())
            return []
    
    def cleanup(self):
        """清理资源"""
        if self._camera_subscription:
            try:
                self._get_video_proxy().unsubscribe(self._camera_subscription)
                self._camera_subscription = None
            except:
                pass
        super(VisionAgent, self).cleanup() 