import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob
import argparse
from moviepy.editor import VideoFileClip
from collections import deque

class LaneDetectionSystem:
    def __init__(self, calibration_images_path='camera_cal', test_images_path='test_images', 
                 output_path='output_images', video_output_path='output_videos', 
                 debug_mode=False):
        # 初始化参数
        self.calibration_images_path = calibration_images_path
        self.test_images_path = test_images_path
        self.output_path = output_path
        self.video_output_path = video_output_path
        self.debug_mode = debug_mode
        
        # 相机校准参数
        self.mtx = None
        self.dist = None
        
        # 透视变换参数
        self.M = None
        self.Minv = None
        
        # 滑动窗口参数
        self.nwindows = 9
        self.margin = 100
        self.minpix = 50
        
        # 车道线拟合参数
        self.left_fit = None
        self.right_fit = None
        self.left_fit_m = None
        self.right_fit_m = None
        
        # 车道线检测历史记录
        self.left_fit_history = deque(maxlen=10)
        self.right_fit_history = deque(maxlen=10)
        
        # 测量参数
        self.ploty = None
        self.left_fitx = None
        self.right_fitx = None
        
        # 车道线曲率和车辆位置
        self.left_curverad = None
        self.right_curverad = None
        self.vehicle_position = None
        
        # 定义转换为米的参数
        self.ym_per_pix = 30/720  # 米每像素y方向
        self.xm_per_pix = 3.7/700  # 米每像素x方向
        
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(video_output_path, exist_ok=True)

    def calibrate_camera(self, nx=9, ny=6):
        """相机校准，计算相机内参和畸变系数"""
        # 准备对象点，如 (0,0,0), (1,0,0), (2,0,0) ..., (8,5,0)
        objp = np.zeros((ny*nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

        # 存储对象点和图像点的数组
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # 读取所有校准图像
        images = glob.glob(f'{self.calibration_images_path}/*.jpg')

        # 遍历图像，查找角点
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 查找棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            # 如果找到，添加对象点，图像点
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # 可选：在图像上绘制角点
                if self.debug_mode:
                    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                    write_name = f'{self.output_path}/corners_found_{idx}.jpg'
                    cv2.imwrite(write_name, img)

        # 进行相机校准
        img_size = (img.shape[1], img.shape[0])
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        
        return ret

    def undistort_image(self, img):
        """使用校准参数校正图像畸变"""
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def apply_thresholds(self, img, s_thresh=(170, 255), sx_thresh=(20, 100)):
        """应用颜色和梯度阈值处理图像"""
        # 转换为HLS颜色空间
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 计算x方向的Sobel梯度
        sobelx = cv2.Sobel(gray, cv2.CV2_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        
        # 应用阈值
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
        
        # 应用S通道阈值
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        
        # 合并阈值结果
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        
        return combined_binary

    def perspective_transform(self, img):
        """执行透视变换，获取鸟瞰图"""
        img_size = (img.shape[1], img.shape[0])
        
        # 定义源点和目标点
        src = np.float32([
            [585, 460],
            [203, 720],
            [1127, 720],
            [695, 460]
        ])
        
        dst = np.float32([
            [320, 0],
            [320, 720],
            [960, 720],
            [960, 0]
        ])
        
        # 计算透视变换矩阵和逆矩阵
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
        
        # 执行透视变换
        warped = cv2.warpPerspective(img, self.M, img_size, flags=cv2.INTER_LINEAR)
        
        return warped

    def find_lane_pixels(self, binary_warped):
        """使用滑动窗口法找到车道线像素"""
        # 计算直方图
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        
        # 创建输出图像用于可视化
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        
        # 找到直方图的左右峰值作为左右车道线的起点
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # 设置窗口高度
        window_height = np.int(binary_warped.shape[0]//self.nwindows)
        
        # 获取图像中所有非零像素的坐标
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # 当前左右车道线的位置
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        # 创建空列表来接收左右车道线像素索引
        left_lane_inds = []
        right_lane_inds = []
        
        # 滑动窗口
        for window in range(self.nwindows):
            # 确定窗口边界的y坐标
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            
            # 确定窗口边界的x坐标
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            
            # 绘制窗口
            if self.debug_mode:
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), 
                              (win_xleft_high, win_y_high), (0,255,0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low), 
                              (win_xright_high, win_y_high), (0,255,0), 2)
            
            # 识别窗口内的非零像素
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            # 添加这些索引到列表中
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # 如果找到了足够的像素，更新下一个窗口的位置
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
        # 连接索引
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # 避免在没有找到车道线时出错
            pass
        
        # 提取左右车道线像素位置
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        return leftx, lefty, rightx, righty, out_img

    def fit_polynomial(self, binary_warped):
        """拟合多项式曲线到车道线像素"""
        # 找到车道线像素
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(binary_warped)
        
        # 拟合二次多项式
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        
        # 生成用于绘制的y值
        self.ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        
        try:
            # 计算拟合曲线上的x值
            self.left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
            self.right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]
        except TypeError:
            # 避免在没有找到车道线时出错
            print('无法拟合车道线')
            self.left_fitx = 1*self.ploty**2 + 1*self.ploty
            self.right_fitx = 1*self.ploty**2 + 1*self.ploty
        
        # 可视化
        if self.debug_mode:
            # 在输出图像上绘制点
            out_img[lefty, leftx] = [255, 0, 0]
            out_img[righty, rightx] = [0, 0, 255]
            
            # 绘制拟合曲线
            plt.plot(self.left_fitx, self.ploty, color='yellow')
            plt.plot(self.right_fitx, self.ploty, color='yellow')
            
        return out_img

    def fit_poly(self, leftx, lefty, rightx, righty):
        """拟合多项式曲线"""
        # 拟合二次多项式
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        
        # 添加到历史记录
        self.left_fit_history.append(self.left_fit)
        self.right_fit_history.append(self.right_fit)
        
        # 平滑处理：使用历史平均值
        self.left_fit = np.mean(self.left_fit_history, axis=0)
        self.right_fit = np.mean(self.right_fit_history, axis=0)
        
        # 生成用于绘制的y值
        self.ploty = np.linspace(0, 719, 720)
        
        # 计算拟合曲线上的x值
        self.left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]
        
        return self.left_fitx, self.right_fitx, self.ploty

    def search_around_poly(self, binary_warped):
        """使用先前的拟合结果搜索新的车道线像素"""
        # 获取图像中非零像素的坐标
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # 设置搜索边距
        margin = self.margin
        
        # 使用先前的多项式拟合结果定义搜索窗口
        left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + 
                        self.left_fit[2] - margin)) & (nonzerox < (self.left_fit[0]*(nonzeroy**2) + 
                        self.left_fit[1]*nonzeroy + self.left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + 
                         self.right_fit[2] - margin)) & (nonzerox < (self.right_fit[0]*(nonzeroy**2) + 
                         self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)))
        
        # 提取左右车道线像素位置
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        # 检查是否找到了足够的点
        if len(leftx) < 50 or len(rightx) < 50:
            # 如果没有找到足够的点，使用滑动窗口重新检测
            return self.fit_polynomial(binary_warped)
        
        # 拟合新的多项式
        self.left_fitx, self.right_fitx, self.ploty = self.fit_poly(leftx, lefty, rightx, righty)
        
        # 创建可视化图像
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        
        # 绘制搜索窗口
        left_line_window1 = np.array([np.transpose(np.vstack([self.left_fitx-margin, self.ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx+margin, 
                                  self.ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        
        right_line_window1 = np.array([np.transpose(np.vstack([self.right_fitx-margin, self.ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx+margin, 
                                   self.ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        
        # 在图像上绘制窗口
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
        # 在图像上绘制检测到的像素点
        result[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        result[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        return result

    def calculate_curvature(self):
        """计算车道线曲率和车辆位置"""
        # 选择图像底部的y值（最接近车辆的位置）
        y_eval = np.max(self.ploty)
        
        # 计算以像素为单位的曲率半径
        self.left_curverad = ((1 + (2*self.left_fit[0]*y_eval + self.left_fit[1])**2)**1.5) / np.absolute(2*self.left_fit[0])
        self.right_curverad = ((1 + (2*self.right_fit[0]*y_eval + self.right_fit[1])**2)**1.5) / np.absolute(2*self.right_fit[0])
        
        # 计算以米为单位的曲率半径
        # 首先拟合以米为单位的多项式
        self.left_fit_m = np.polyfit(self.ploty*self.ym_per_pix, self.left_fitx*self.xm_per_pix, 2)
        self.right_fit_m = np.polyfit(self.ploty*self.ym_per_pix, self.right_fitx*self.xm_per_pix, 2)
        
        # 计算以米为单位的曲率
        left_curverad_m = ((1 + (2*self.left_fit_m[0]*y_eval*self.ym_per_pix + self.left_fit_m[1])**2)**1.5) / np.absolute(2*self.left_fit_m[0])
        right_curverad_m = ((1 + (2*self.right_fit_m[0]*y_eval*self.ym_per_pix + self.right_fit_m[1])**2)**1.5) / np.absolute(2*self.right_fit_m[0])
        
        # 计算车辆位置
        lane_center = (self.left_fitx[-1] + self.right_fitx[-1]) / 2
        image_center = 1280 / 2
        self.vehicle_position = (image_center - lane_center) * self.xm_per_pix
        
        return left_curverad_m, right_curverad_m, self.vehicle_position

    def draw_lane(self, undist, binary_warped):
        """在原始图像上绘制检测到的车道"""
        # 创建空白图像用于绘制车道
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # 填充车道区域
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # 在空白图像上绘制车道区域
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        
        # 绘制车道线
        cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255, 0, 0), thickness=20)
        cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0, 0, 255), thickness=20)
        
        # 执行逆透视变换
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (undist.shape[1], undist.shape[0]))
        
        # 将结果与原始图像合并
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        
        # 计算并添加曲率和位置信息
        left_curverad, right_curverad, vehicle_position = self.calculate_curvature()
        
        # 添加曲率信息
        curvature_text = f"Radius of Curvature = {int((left_curverad + right_curverad)/2)}m"
        cv2.putText(result, curvature_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 添加车辆位置信息
        position_text = f"Vehicle is {abs(vehicle_position):.2f}m {'left' if vehicle_position < 0 else 'right'} of center"
        cv2.putText(result, position_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return result

    def process_image(self, img):
        """处理单张图像的完整流程"""
        # 校正图像畸变
        undistorted = self.undistort_image(img)
        
        # 应用阈值处理
        binary = self.apply_thresholds(undistorted)
        
        # 执行透视变换
        warped = self.perspective_transform(binary)
        
        # 检测车道线
        if self.left_fit is None or self.right_fit is None:
            # 首次检测或丢失车道线时，使用滑动窗口法
            self.fit_polynomial(warped)
        else:
            # 否则使用基于先前拟合结果的搜索
            self.search_around_poly(warped)
        
        # 在原始图像上绘制车道
        result = self.draw_lane(undistorted, warped)
        
        return result

    def process_test_images(self):
        """处理测试图像"""
        # 确保相机已校准
        if self.mtx is None or self.dist is None:
            self.calibrate_camera()
        
        # 获取测试图像列表
        images = glob.glob(f'{self.test_images_path}/*.jpg')
        
        # 处理每张图像
        for idx, fname in enumerate(images):
            print(f"处理图像: {fname}")
            
            # 读取图像
            img = mpimg.imread(fname)
            
            # 处理图像
            result = self.process_image(img)
            
            # 保存结果
            save_path = f"{self.output_path}/{os.path.basename(fname)}"
            mpimg.imsave(save_path, result)
            
            if self.debug_mode:
                # 显示结果
                plt.figure(figsize=(10, 8))
                plt.imshow(result)
                plt.show()

    def process_video(self, video_path, output_path):
        """处理视频"""
        # 确保相机已校准
        if self.mtx is None or self.dist is None:
            self.calibrate_camera()
        
        # 加载视频
        clip = VideoFileClip(video_path)
        
        # 处理视频帧
        processed_clip = clip.fl_image(self.process_image)
        
        # 保存结果
        processed_clip.write_videofile(output_path, audio=False)

def main():
    parser = argparse.ArgumentParser(description='车道线检测系统')
    parser.add_argument('--test_images', action='store_true', help='处理测试图像')
    parser.add_argument('--video', type=str, help='处理视频文件')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    # 初始化车道线检测系统
    system = LaneDetectionSystem(debug_mode=args.debug)
    
    # 根据参数执行相应操作
    if args.test_images:
        system.process_test_images()
    elif args.video:
        video_name = os.path.basename(args.video)
        output_path = f"output_videos/processed_{video_name}"
        system.process_video(args.video, output_path)
    else:
        print("请指定要执行的操作：--test_images 或 --video [视频路径]")

if __name__ == "__main__":
    main()    
