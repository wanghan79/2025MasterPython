# 用于将视频切割为画面和音频
import os
import cv2
import uuid
from pydub import AudioSegment
from .transcriber import Transcriber
import ffmpeg

class MediaProcessor:
    """
    媒体处理器类
    负责处理视频上传、音频提取、转录等媒体相关操作
    """
    
    def __init__(self, upload_folder):
        """
        初始化媒体处理器
        
        参数:
            upload_folder (str): 上传文件的根目录路径
        """
        self.upload_folder = upload_folder
        # 设置视频、音频和转录文件的存储路径
        self.video_folder = os.path.join(upload_folder, 'videos')
        self.audio_folder = os.path.join(upload_folder, 'audios')
        self.transcript_folder = os.path.join(upload_folder, 'transcripts')
        
        # 创建必要的目录结构
        os.makedirs(self.video_folder, exist_ok=True)
        os.makedirs(self.audio_folder, exist_ok=True)
        os.makedirs(self.transcript_folder, exist_ok=True)
        
        # 初始化转录器实例
        self.transcriber = Transcriber()
    
    def save_video(self, video_file):
        """
        保存上传的视频文件
        
        参数:
            video_file (FileStorage): Flask上传的视频文件对象
            
        返回:
            str: 生成的唯一文件名
        """
        # 使用UUID生成唯一文件名，保留原始文件扩展名
        filename = str(uuid.uuid4()) + os.path.splitext(video_file.filename)[1]
        video_path = os.path.join(self.video_folder, filename)
        video_file.save(video_path)
        return filename
    
    def extract_audio(self, video_filename):
        """
        从视频文件中提取音频
        
        参数:
            video_filename (str): 视频文件名
            
        返回:
            str: 生成的音频文件名
            
        异常:
            Exception: 当音频提取失败时抛出
        """
        video_path = os.path.join(self.video_folder, video_filename)
        audio_filename = os.path.splitext(video_filename)[0] + '.wav'
        audio_path = os.path.join(self.audio_folder, audio_filename)
        
        try:
            # 使用pydub库提取音频并保存为WAV格式
            audio = AudioSegment.from_file(video_path)
            audio.export(audio_path, format="wav")
            return audio_filename
        except Exception as e:
            raise Exception(f"音频提取失败: {str(e)}")
    
    def transcribe_audio(self, audio_filename):
        """
        转录音频文件为文本
        
        参数:
            audio_filename (str): 音频文件名
            
        返回:
            str: 转录结果文件名
            
        异常:
            Exception: 当转录失败时抛出
        """
        audio_path = os.path.join(self.audio_folder, audio_filename)
        try:
            transcript_file = self.transcriber.transcribe(audio_path, self.transcript_folder)
            return os.path.basename(transcript_file)
        except Exception as e:
            raise Exception(f"转录失败: {str(e)}")
    
    def get_video_info(self, video_filename):
        """
        获取视频文件的基本信息
        
        参数:
            video_filename (str): 视频文件名
            
        返回:
            dict: 包含视频时长、大小、帧率和分辨率的字典
        """
        video_path = os.path.join(self.video_folder, video_filename)
        
        try:
            # 使用ffmpeg获取视频信息
            probe = ffmpeg.probe(video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            
            # 获取准确的视频时长
            duration = float(probe['format']['duration'])
            
            # 获取其他视频属性
            width = int(video_info['width'])
            height = int(video_info['height'])
            fps = eval(video_info['r_frame_rate'])  # 处理分数形式的帧率
            
            # 获取文件大小
            size = os.path.getsize(video_path)
            
            return {
                'duration': duration,
                'size': size,
                'fps': fps,
                'resolution': f"{width}x{height}"
            }
        except Exception as e:
            raise Exception(f"获取视频信息失败: {str(e)}") 