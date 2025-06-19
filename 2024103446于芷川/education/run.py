# 设置OpenMP环境变量
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from app.utils.media_processor import MediaProcessor
from werkzeug.utils import secure_filename
import json
import whisper
import cv2
from pydub import AudioSegment
import warnings
import logging
import uuid
import threading
from app.utils.emotion_analyzer import EmotionAnalyzer

# 禁用特定的 FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning, module='whisper')

# 设置日志级别为 ERROR，只显示错误信息
logging.getLogger('whisper').setLevel(logging.ERROR)

# 全局变量，用于存储情感分析器实例
global_emotion_analyzer = None

def get_emotion_analyzer():
    global global_emotion_analyzer
    if global_emotion_analyzer is None:
        global_emotion_analyzer = EmotionAnalyzer()
    return global_emotion_analyzer

app = Flask(__name__, 
    template_folder='app/templates',
    static_folder='app/static')

# 配置上传文件夹
current_dir = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(current_dir, 'app', 'uploads')
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 最大文件大小
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 确保所有必要的上传目录存在
for subdir in ['videos', 'audios', 'transcripts']:
    dir_path = os.path.join(app.config['UPLOAD_FOLDER'], subdir)
    os.makedirs(dir_path, exist_ok=True)
    logger.info(f"创建目录: {dir_path}")

# 打印当前使用的上传路径
logger.info(f"使用上传目录: {app.config['UPLOAD_FOLDER']}")

# 初始化媒体处理器
media_processor = MediaProcessor(app.config['UPLOAD_FOLDER'])

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'flv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_audio(video_path, audio_path):
    """从视频中提取音频"""
    try:
        # 使用pydub提取音频
        video = AudioSegment.from_file(video_path)
        video.export(audio_path, format="wav")
    except Exception as e:
        raise Exception(f"音频提取失败: {str(e)}")

def get_video_info(video_path):
    """获取视频信息"""
    try:
        import ffmpeg
        
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
            'resolution': f'{width}x{height}',
            'fps': fps,
            'size': size
        }
    except Exception as e:
        raise Exception(f"获取视频信息失败: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/videos')
def videos():
    return render_template('videos.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        logger.error('没有上传文件')
        return jsonify({'error': '没有上传文件'}), 400
        
    video = request.files['video']
    
    if video.filename == '':
        logger.error('没有选择文件')
        return jsonify({'error': '没有选择文件'}), 400
        
    if not allowed_file(video.filename):
        logger.error(f'不支持的文件格式: {video.filename}')
        return jsonify({'error': '不支持的文件格式'}), 400
    
    try:
        # 保存视频文件
        video_filename = secure_filename(video.filename)
        videos_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'videos')
        video_path = os.path.join(videos_dir, video_filename)
        
        logger.info(f"保存视频文件到: {video_path}")
        video.save(video_path)
        
        if not os.path.exists(video_path):
            logger.error(f"视频文件保存失败，文件不存在: {video_path}")
            return jsonify({'error': '视频文件保存失败'}), 500
            
        logger.info(f"视频文件大小: {os.path.getsize(video_path)} 字节")
        
        # 提取音频
        audio_filename = os.path.splitext(video_filename)[0] + '.wav'
        audios_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'audios')
        audio_path = os.path.join(audios_dir, audio_filename)
        
        logger.info(f"开始提取音频到: {audio_path}")
        extract_audio(video_path, audio_path)
        
        if not os.path.exists(audio_path):
            logger.error(f"音频提取失败，文件不存在: {audio_path}")
            return jsonify({'error': '音频提取失败'}), 500
            
        logger.info(f"音频文件大小: {os.path.getsize(audio_path)} 字节")
        
        # 获取视频信息
        video_info = get_video_info(video_path)
        logger.info(f"视频信息: {video_info}")
        
        return jsonify({
            'success': True,
            'video_filename': video_filename,
            'audio_filename': audio_filename,
            'video_info': video_info
        })
    
    except Exception as e:
        logger.error(f"处理上传文件时出错: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# 存储转录任务状态
transcription_tasks = {}

@app.route('/transcribe', methods=['POST'])
def transcribe_video():
    try:
        data = request.get_json()
        video_filename = data.get('video_filename')
        if not video_filename:
            return jsonify({'error': '未提供视频文件名'}), 400
            
        audio_filename = os.path.splitext(video_filename)[0] + '.wav'
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'audios', audio_filename)
        
        # 创建转录任务状态
        task_id = str(uuid.uuid4())
        transcription_tasks[task_id] = {
            'status': 'processing',
            'progress': 0
        }
        
        # 在后台线程中执行转录
        def run_transcription():
            try:
                # 使用Whisper进行转录
                # - Whisper模型: base（可选：tiny, base, small, medium, large）
                model = whisper.load_model("tiny")
                initial_prompt = "以下是普通话的句子，内容关于教育。请准确识别，避免重复。"
                
                # 更新进度
                transcription_tasks[task_id]['progress'] = 30
                
                result = model.transcribe(
                    audio_path,
                    language='zh',
                    initial_prompt=initial_prompt,
                    no_speech_threshold=0.6,
                    logprob_threshold=-1.0,
                    condition_on_previous_text=True,
                    temperature=0.0
                )
                
                # 更新进度
                transcription_tasks[task_id]['progress'] = 80
                
                # 保存转录结果
                transcript_filename = os.path.splitext(video_filename)[0] + '.json'
                transcript_path = os.path.join(app.config['UPLOAD_FOLDER'], 'transcripts', transcript_filename)
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                # 完成
                transcription_tasks[task_id]['status'] = 'completed'
                transcription_tasks[task_id]['progress'] = 100
                transcription_tasks[task_id]['result'] = result
                
            except Exception as e:
                transcription_tasks[task_id]['status'] = 'error'
                transcription_tasks[task_id]['error'] = str(e)
        
        # 启动后台线程
        thread = threading.Thread(target=run_transcription)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'task_id': task_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/transcribe/status/<task_id>')
def get_transcription_status(task_id):
    if task_id not in transcription_tasks:
        return jsonify({'error': '任务不存在'}), 404
        
    task = transcription_tasks[task_id]
    response = {
        'status': task['status'],
        'progress': task['progress']
    }
    
    if task['status'] == 'completed':
        response['result'] = task['result']
    elif task['status'] == 'error':
        response['error'] = task['error']
        
    return jsonify(response)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/uploads/videos/<path:filename>')
def uploaded_video(filename):
    """访问上传的视频文件"""
    logger.info(f"请求访问视频文件: {filename}")
    videos_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'videos')
    logger.info(f"视频目录: {videos_dir}")
    return send_from_directory(videos_dir, filename)

@app.route('/uploads/audios/<path:filename>')
def uploaded_audio(filename):
    """访问上传的音频文件"""
    logger.info(f"请求访问音频文件: {filename}")
    audios_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'audios')
    logger.info(f"音频目录: {audios_dir}")
    return send_from_directory(audios_dir, filename)

@app.route('/uploads/transcripts/<path:filename>')
def uploaded_transcript(filename):
    """访问转录文本文件"""
    logger.info(f"请求访问转录文件: {filename}")
    transcripts_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'transcripts')
    logger.info(f"转录目录: {transcripts_dir}")
    return send_from_directory(transcripts_dir, filename)

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    try:
        data = request.get_json()
        video_filename = data.get('video_filename')
        if not video_filename:
            logger.error('情感分析请求中未提供视频文件名')
            return jsonify({'success': False, 'error': '未提供视频文件名'}), 400
            
        # 获取音频和转录文本文件路径
        audio_filename = os.path.splitext(video_filename)[0] + '.wav'
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'audios', audio_filename)
        
        transcript_filename = os.path.splitext(video_filename)[0] + '.json'
        transcript_path = os.path.join(app.config['UPLOAD_FOLDER'], 'transcripts', transcript_filename)
        
        logger.info(f'开始情感分析 - 视频: {video_filename}')
        logger.info(f'音频路径: {audio_path}')
        logger.info(f'转录文件路径: {transcript_path}')
        
        # 检查文件是否存在
        if not os.path.exists(audio_path):
            logger.error(f'音频文件不存在: {audio_path}')
            return jsonify({'success': False, 'error': '找不到音频文件'}), 404
            
        if not os.path.exists(transcript_path):
            logger.error(f'转录文件不存在: {transcript_path}')
            return jsonify({'success': False, 'error': '找不到转录文件'}), 404
        
        # 读取转录文本
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
                text = transcript_data.get('text', '')
                if not text:
                    logger.error('转录文本为空')
                    return jsonify({'success': False, 'error': '转录文本为空'}), 400
        except json.JSONDecodeError as e:
            logger.error(f'转录文件解析失败: {str(e)}')
            return jsonify({'success': False, 'error': '转录文件格式错误'}), 400
        
        logger.info('开始进行情感分析...')
        # 使用全局情感分析器实例
        emotion_analyzer = get_emotion_analyzer()
        result = emotion_analyzer.analyze_emotion(audio_path, text)
        logger.info('情感分析完成')
        logger.info(f'分析结果: {result}')
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except FileNotFoundError as e:
        logger.error(f'文件未找到: {str(e)}')
        return jsonify({'success': False, 'error': '找不到所需文件'}), 404
    except Exception as e:
        logger.error(f'情感分析失败: {str(e)}')
        return jsonify({'success': False, 'error': f'情感分析失败: {str(e)}'}), 500

if __name__ == '__main__':
    # 在启动服务器前初始化情感分析器
    get_emotion_analyzer()
    # 关闭Flask的重新加载功能
    app.run(debug=True, use_reloader=False) 