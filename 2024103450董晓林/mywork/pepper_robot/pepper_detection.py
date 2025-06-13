#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import requests
import base64
import json
import numpy as np
import sys
import os
import time
import traceback
import math

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from pepper_client.api import PepperRobotClient

def load_calibration_data():
    """加载相机标定数据
    
    Returns:
        dict: 标定数据，包含像素到米的转换比例等信息
    """
    calibration_file = os.path.join(current_dir, 'camera_calibration.json')
    if not os.path.exists(calibration_file):
        print("警告：未找到标定数据文件，将使用默认值")
        return {
            'average_pixel_to_meter': 0.0005,  # 默认值
            'calibration_data': []
        }
    
    try:
        with open(calibration_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载标定数据失败: {e}")
        return {
            'average_pixel_to_meter': 0.0005,  # 默认值
            'calibration_data': []
        }

def estimate_distance(height_px, calibration_data):
    """根据物体像素高度和标定数据估算相机到物体的距离Z（单位：cm）
    
    Args:
        height_px (float): 物体在图像中的像素高度
        calibration_data (dict): 标定数据
        
    Returns:
        float: 估算的距离（厘米）
    """
    if height_px == 0:
        print("警告：物体像素高度为0，无法计算距离")
        return None
        
    print(f"\n=== 距离计算信息 ===")
    print(f"物体像素高度: {height_px} 像素")
    
    # 如果没有标定数据，使用默认值
    if not calibration_data['calibration_data']:
        print("使用默认焦距计算距离")
        default_distance = (525.0 * 20.0) / height_px
        print(f"默认计算距离: {default_distance:.2f} cm")
        return default_distance
        
    # 使用标定数据计算距离
    # 找到最接近的标定数据点
    closest_data = min(
        calibration_data['calibration_data'],
        key=lambda x: abs(x['object_pixels'] - height_px)
    )
    
    print(f"最接近的标定数据:")
    print(f"- 标定距离: {closest_data['distance_cm']} cm")
    print(f"- 标定像素: {closest_data['object_pixels']} 像素")
    print(f"- 实际尺寸: {closest_data['real_size_cm']} cm")
    
    # 使用相似三角形原理计算距离
    # 根据标定数据计算焦距
    focal_length = (closest_data['object_pixels'] * closest_data['distance_cm']) / closest_data['real_size_cm']
    print(f"计算得到的焦距: {focal_length:.2f} 像素")
    
    # 使用焦距计算实际距离
    calculated_distance = (focal_length * closest_data['real_size_cm']) / height_px
    print(f"计算得到的距离: {calculated_distance:.2f} cm")
    
    return calculated_distance

def calculate_camera_offset(image, center_x, center_y, Z, camera_type='bottom', calibration_data=None):
    """计算物体在相机坐标系下的位移，并转换到Torso坐标系
    
    Args:
        image (numpy.ndarray): 图像数据
        center_x (int): 物体中心点x坐标
        center_y (int): 物体中心点y坐标
        Z (float): 物体到相机的距离（厘米）
        camera_type (str): 相机类型，'top' 或 'bottom'
        calibration_data (dict): 标定数据
        
    Returns:
        tuple: (dx, dy, dz) Torso坐标系下的位移（米）
    """
    # 图像中心坐标
    img_h, img_w = image.shape[:2]
    image_cx = img_w // 2
    image_cy = img_h // 2
    
    # 物体偏移（像素）
    dx_pixel = center_x - image_cx   # 右偏为正
    dy_pixel = center_y - image_cy   # 下偏为正
    
    # 使用标定数据中的像素到米转换比例
    if calibration_data and calibration_data['calibration_data']:
        pixel_to_meter = calibration_data['average_pixel_to_meter']
    else:
        # 如果没有标定数据，使用默认值
        fov_h = 58.4  # 水平视场角（度）
        fov_v = 45.5  # 垂直视场角（度）
        theta_h = (fov_h / 2) * (math.pi / 180)
        theta_v = (fov_v / 2) * (math.pi / 180)
        width_at_1m = 2 * math.tan(theta_h)
        height_at_1m = 2 * math.tan(theta_v)
        pixel_to_meter_h = width_at_1m / img_w
        pixel_to_meter_v = height_at_1m / img_h
        pixel_to_meter = (pixel_to_meter_h + pixel_to_meter_v) / 2
    
    # 转换为相机坐标系下的相对位移（单位：米）
    dx_camera = dx_pixel * pixel_to_meter
    dy_camera = dy_pixel * pixel_to_meter
    dz_camera = Z / 100.0  # Z是cm，转换为米
    
    # 将数据写入临时文件
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    input_file = os.path.join(temp_dir, 'transform_input.json')
    output_file = os.path.join(temp_dir, 'transform_output.json')
    
    # 写入输入数据
    with open(input_file, 'w') as f:
        json.dump({
            'dx_camera': dx_camera,
            'dy_camera': dy_camera,
            'dz_camera': dz_camera,
            'camera_type': camera_type
        }, f)
    
    # 等待输出文件生成
    max_wait = 10  # 最大等待时间（秒）
    start_time = time.time()
    
    while not os.path.exists(output_file):
        if time.time() - start_time > max_wait:
            print("等待坐标转换超时，返回相机坐标系下的位置")
            return dx_camera, dy_camera, dz_camera
        time.sleep(0.1)
    
    # 读取输出数据
    try:
        with open(output_file, 'r') as f:
            result = json.load(f)
            position_torso = (result['dx_torso'], result['dy_torso'], result['dz_torso'])
            
            if client.debug:
                print(f"\n=== 坐标转换结果 ===")
                print(f"源坐标系: {camera_type}")
                print(f"目标坐标系: Torso")
                print(f"相机坐标系位置: ({dx_camera:.3f}, {dy_camera:.3f}, {dz_camera:.3f}) m")
                print(f"Torso坐标系位置: ({position_torso[0]:.3f}, {position_torso[1]:.3f}, {position_torso[2]:.3f}) m")
                print(f"使用的像素到米转换比例: {pixel_to_meter:.6f}")
            
            return position_torso
    except Exception as e:
        print(f"读取转换结果失败: {e}")
        return dx_camera, dy_camera, dz_camera

class PepperPhotoDetector:
    def __init__(self, api_key, secret_key, host="localhost", port=5555, camera_type='bottom'):
        """初始化Pepper拍照和物体检测器
        
        Args:
            api_key (str): 百度AI平台的API Key
            secret_key (str): 百度AI平台的Secret Key
            host (str): Pepper机器人服务器地址
            port (int): Pepper机器人服务器端口
            camera_type (str): 相机类型，'top' 或 'bottom'
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.client = PepperRobotClient(host=host, port=port, debug=True)
        self.camera_type = camera_type
        self.access_token = self._get_access_token()
        self.last_detection = None  # 存储最后一次检测结果
        self.calibration_data = load_calibration_data()  # 加载标定数据
        
    def _get_access_token(self):
        """获取百度云 access_token"""
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key
        }
        res = requests.post(url, data=params)
        return res.json().get("access_token")
        
    def take_photo(self, save_path=None):
        """拍照并保存
        
        Args:
            save_path (str): 保存路径，如果为None则不保存
            
        Returns:
            numpy.ndarray: 图像数据
        """
        # 检查连接
        if not self.client.check_connection():
            print("无法连接到机器人服务器")
            return None
            
        # 拍照
        result = self.client.take_picture(
            camera_id=1,    # 0为顶部相机，1为底部相机
            resolution=2,   # 2=VGA分辨率
            color_space=11  # 11=BGR颜色空间
        )
        
        if result is None:
            print("拍照失败")
            return None
            
        # 从结果中提取图像数据
        width = result['width']
        height = result['height']
        channels = result['channels']
        image_data = np.array(result['data'], dtype=np.uint8)
        
        # 重塑图像数组
        if channels == 1:
            image = image_data.reshape((height, width))
        else:
            image = image_data.reshape((height, width, channels))
            
        # 保存图片
        if save_path:
            cv2.imwrite(save_path, image)
            print(f"图片已保存到: {save_path}")
            
        return image
        
    def detect_objects(self, image):
        """调用百度AI平台进行通用物体检测
        
        Args:
            image (numpy.ndarray): 图像数据
            
        Returns:
            dict: 检测结果，包含物体的边界框和置信度
        """
        # 将图像转换为base64编码
        _, img_encoded = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        # 构建请求URL和参数
        request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/object_detect"
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        params = {
            "image": img_base64,
            "with_face": 0,  # 不检测人脸
            "baike_num": 0,  # 不返回百科信息
            "top_num": 1     # 返回置信度最高的1个结果
        }
        
        try:
            # 发送请求
            response = requests.post(
                f"{request_url}?access_token={self.access_token}",
                data=params,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print("API返回结果:", json.dumps(result, indent=2, ensure_ascii=False))
                self.last_detection = result  # 保存检测结果
                return result
            else:
                print("物体检测请求失败，状态码：", response.status_code)
                return None
                
        except Exception as e:
            print(f"物体检测失败: {e}")
            return None
            
    def classify_image(self, image):
        """调用百度AI平台进行图像分类
        
        Args:
            image (numpy.ndarray): 图像数据
            
        Returns:
            dict: 分类结果
        """
        # 将图像转换为base64编码
        _, img_encoded = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        # 构建请求URL和参数
        request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v2/advanced_general"
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        params = {"image": img_base64}
        
        try:
            # 发送请求
            response = requests.post(
                f"{request_url}?access_token={self.access_token}",
                data=params,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print("图像分类结果:", json.dumps(result, indent=2, ensure_ascii=False))
                return result
            else:
                print("图像分类请求失败，状态码：", response.status_code)
                return None
                
        except Exception as e:
            print(f"图像分类失败: {e}")
            return None
            
    def visualize_results(self, image, detection_result, save_path=None):
        """可视化检测结果并估算距离
        
        Args:
            image (numpy.ndarray): 原始图像
            detection_result (dict): 物体检测结果
            save_path (str): 保存路径，如果为None则不保存
            
        Returns:
            numpy.ndarray: 标注后的图像
        """
        if detection_result is None:
            print("检测结果为空")
            return image
            
        # 复制图像
        vis_image = image.copy()
        
        # 解析检测结果
        if isinstance(detection_result, dict):
            # 获取检测结果
            result = detection_result.get('result', {})
            
            if result:
                # 获取边界框坐标
                top = int(result.get('top', 0))
                left = int(result.get('left', 0))
                width = int(result.get('width', 0))
                height = int(result.get('height', 0))
                
                # 计算物体中心点坐标
                center_x = left + (width // 2)
                center_y = top + (height // 2)
                
                # 估算距离
                Z = estimate_distance(height_px=height, calibration_data=self.calibration_data)
                
                # 计算相机坐标系下的位移，并转换到Torso坐标系
                dx, dy, dz = calculate_camera_offset(
                    image, 
                    center_x, 
                    center_y, 
                    Z,
                    camera_type=self.camera_type,
                    calibration_data=self.calibration_data
                )
                
                # 保存位移信息供后续使用
                self.last_detection = {
                    'position': (dx, dy, dz),
                    'bounding_box': (top, left, width, height)
                }
                
                # 打印检测信息
                print(f"\n=== 检测到的物体边界框 ===")
                print(f"top: {top}, left: {left}, width: {width}, height: {height}")
                print(f"中心点: ({center_x}, {center_y})")
                print(f"估算的距离 Z ≈ {Z:.2f} cm")
                print(f"\n=== Torso坐标系下的位移 ===")
                print(f"dx ≈ {dx:.3f} m (右偏为正)")
                print(f"dy ≈ {dy:.3f} m (下偏为正)")
                print(f"dz ≈ {dz:.3f} m (远离为正)")
                
                # 1. 绘制绿色边界框
                cv2.rectangle(vis_image, (left, top), (left+width, top+height), (0, 255, 0), 2)
                
                # 2. 在物体中心点绘制红色圆点
                cv2.circle(vis_image, (center_x, center_y), 8, (0, 0, 255), -1)
                
                # 3. 在物体中心点上方显示坐标
                coord_text = f"({center_x}, {center_y})"
                cv2.putText(vis_image, coord_text, (center_x-30, center_y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # 4. 在物体边界框下方显示距离信息
                distance_text = f"Z ≈ {Z:.1f} cm"
                cv2.putText(vis_image, distance_text, (left, top + height + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # 5. 在图像右侧显示位移信息
                offset_text = f"dx: {dx:.3f}m\ndy: {dy:.3f}m\ndz: {dz:.3f}m"
                cv2.putText(vis_image, offset_text, (image.shape[1] - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        # 保存结果
        if save_path:
            cv2.imwrite(save_path, vis_image)
            print(f"\n结果已保存到: {save_path}")
            
        return vis_image
        
    def move_to_object(self, distance=0.5, speed=0.3):
        """向前移动到指定距离
        
        Args:
            distance (float): 移动距离（米）
            speed (float): 移动速度，范围0.0-1.0
            
        Returns:
            bool: 是否成功
        """
        try:
            print(f"\n开始向前移动 {distance} 米...")
            success = self.client.move_to(distance, 0, 0)
            if success:
                print("移动成功")
                time.sleep(2)  # 等待移动完成
                return True
            else:
                print("移动失败")
                return False
        except Exception as e:
            print(f"移动过程中发生错误: {str(e)}")
            return False
            
    def grab_object(self):
        """执行抓取动作
        
        Returns:
            bool: 是否成功完成抓取
        """
        if self.last_detection is None:
            print("没有检测到物体，无法执行抓取")
            return False
            
        try:
            # 1. 唤醒机器人
            print("\n=== 唤醒机器人 ===")
            if not self.client.wake_up():
                print("唤醒机器人失败")
                return False
                
            # 2. 获取物体位置
            dx, dy, dz = self.last_detection['position']
            print(f"\n=== 物体位置信息 ===")
            print(f"检测到的物体位置: dx={dx:.3f}m, dy={dy:.3f}m, dz={dz:.3f}m")
            
            # 3. 安全检查
            if dz < 0.05:  # 物体太近（5厘米）
                print(f"警告：物体距离太近 ({dz:.3f}米)，可能存在碰撞风险")
                print("建议：将物体稍微放远一些，距离保持在5厘米以上")
                return False
                
            if abs(dy) > 0.3:  # 物体太偏
                print(f"警告：物体位置偏离中心太远 (dy={dy:.3f}米)")
                print("建议：将物体放在机器人正前方")
                return False
            
            # 4. 计算移动距离
            move_distance = dz - 0.2  # 物体距离减去手臂长度（约0.2米）
            print(f"\n=== 移动距离计算 ===")
            print(f"物体距离: {dz:.3f}米")
            print(f"手臂长度: 0.2米")
            print(f"需要移动的距离: {move_distance:.3f}米")
            
            # 5. 抬起手臂到准备位置
            print("\n=== 抬起手臂到准备位置 ===")
            success = self.client.set_arm_angles('R', [-0.2, 0.0, 0.0, 0.8, 0.0], speed=0.2)
            if not success:
                print("抬起手臂失败")
                return False
            print("手臂已抬起")
            time.sleep(2)  # 等待手臂移动完成
            
            # 6. 向前移动到合适位置
            if move_distance > 0:
                print(f"\n=== 向前移动接近物体 ===")
                print(f"开始向前移动 {move_distance:.3f} 米")
                success = self.client.move_to(move_distance, 0, 0)
                if not success:
                    print("移动失败")
                    return False
                print("移动命令已发送")
                time.sleep(3)  # 等待移动完成
                print("移动完成")
            else:
                print("物体距离合适，无需移动")
            
            # 7. 调整手臂到抓取位置
            print("\n=== 调整手臂到抓取位置 ===")
            success = self.client.set_arm_angles('R', [-0.3, 0.0, 0.0, 0.9, 0.0], speed=0.2)
            if not success:
                print("调整手臂位置失败")
                return False
            print("手臂已调整到抓取位置")
            time.sleep(1)  # 等待手臂调整完成
            
            # 8. 张开手爪
            print("\n=== 张开手爪 ===")
            success = self.client.set_hand_open('R', 0.0, speed=0.2)  # 0.0 表示完全张开
            if not success:
                print("张开手爪失败")
                return False
            print("手爪已张开")
            time.sleep(1)  # 等待手爪张开
            
            # 9. 闭合手爪进行抓取
            print("\n=== 闭合手爪进行抓取 ===")
            success = self.client.set_hand_open('R', 1.0, speed=0.2)  # 1.0 表示完全闭合
            if not success:
                print("闭合手爪失败")
                return False
            print("手爪已闭合")
            time.sleep(1)  # 等待手爪闭合
            
            # 10. 张开手爪释放物体
            print("\n=== 张开手爪释放物体 ===")
            success = self.client.set_hand_open('R', 0.0, speed=0.2)
            if not success:
                print("张开手爪失败")
                return False
            print("手爪已张开")
            time.sleep(1)  # 等待手爪张开
            
            # 11. 向后移动回到原位置
            print("\n=== 向后移动回到原位置 ===")
            if move_distance > 0:
                print(f"开始向后移动 {move_distance:.3f} 米")
                success = self.client.move_to(-move_distance, 0, 0)
                if not success:
                    print("向后移动失败")
                    return False
                print("移动命令已发送")
                time.sleep(3)  # 等待移动完成
                print("已回到原位置")
            
            # 12. 放下手臂
            print("\n=== 放下手臂 ===")
            success = self.client.set_arm_angles('R', [1.5, 0.0, 0.0, 0.0, 0.0], speed=0.2)  # 修改为1.5弧度，约86度
            if not success:
                print("放下手臂失败")
                return False
            print("手臂已放下")
            time.sleep(2)  # 等待手臂放下完成
            
            print("\n抓取操作完成 ✅")
            return True
            
        except Exception as e:
            print(f"\n抓取失败: {e}")
            print(traceback.format_exc())
            # 尝试回到安全位置
            try:
                self.client.set_arm_angles('R', [1.5, 0.0, 0.0, 0.0, 0.0], speed=0.2)
            except:
                pass
            return False
        
    def close(self):
        """关闭连接"""
        self.client.close()

def main():
    # 百度AI平台配置
    API_KEY = "f2CgLYwG5OYQeUEuHAcKZiXz"
    SECRET_KEY = "dbwLF04fpplbbt7jSsOrLNvmC3sB1nqU"
    
    # 创建物体检测器
    detector = PepperPhotoDetector(API_KEY, SECRET_KEY)
    
    try:
        # 拍照
        print("正在拍照...")
        image = detector.take_photo("original.jpg")
        if image is None:
            return
            
        # 物体检测
        print("正在进行物体检测...")
        detection_result = detector.detect_objects(image)
        
        # 可视化结果
        print("正在可视化结果...")
        vis_image = detector.visualize_results(
            image, 
            detection_result, 
            "detection_result.jpg"
        )
        
        # 显示结果
        cv2.imshow("Object Detection", vis_image)
        cv2.waitKey(0)
        
        # 执行抓取动作
        detector.grab_object()
        
    finally:
        detector.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 