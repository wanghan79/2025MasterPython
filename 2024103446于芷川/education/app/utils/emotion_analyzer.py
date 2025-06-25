import librosa
import numpy as np
from transformers import pipeline
import torch
import logging

class EmotionAnalyzer:
    """
    情感分析器类
    结合音频特征和文本内容进行情感分析
    """
    
    def __init__(self):
        """
        初始化情感分析器
        加载必要的模型和设置参数
        """
        # 设置日志级别
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 加载文本情感分析模型
        try:
            self.logger.info("正在加载文本情感分析模型...")
            self.text_analyzer = pipeline(
                "sentiment-analysis",
                model="uer/roberta-base-finetuned-jd-binary-chinese",
                tokenizer="uer/roberta-base-finetuned-jd-binary-chinese"
            )
            self.logger.info("文本情感分析模型加载完成")
        except Exception as e:
            self.logger.error(f"加载文本情感分析模型失败: {str(e)}")
            raise
            
        # 情感标签映射
        self.emotion_labels = {
            'positive': '积极',
            'negative': '消极',
            'neutral': '平静'
        }
        
        # 详细情感类别
        self.detailed_emotions = {
            'positive': ['热情', '愉快', '专注'],
            'negative': ['疲惫', '紧张', '焦虑'],
            'neutral': ['自然', '放松', '沉稳']
        }
    
    def analyze_audio_features(self, audio_path):
        """
        分析音频特征
        
        参数:
            audio_path (str): 音频文件路径
            
        返回:
            dict: 包含音频特征的字典
        """
        try:
            self.logger.info(f"开始分析音频特征: {audio_path}")
            
            # 加载音频文件
            y, sr = librosa.load(audio_path)
            self.logger.info(f"音频加载完成，采样率: {sr}Hz")
            
            # 提取特征
            # 1. 音高特征
            self.logger.info("正在提取音高特征...")
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            
            # 2. 能量特征
            self.logger.info("正在提取能量特征...")
            rmse = librosa.feature.rms(y=y)[0]
            energy_mean = np.mean(rmse)
            energy_std = np.std(rmse)
            
            # 3. 语速特征（过零率）
            self.logger.info("正在提取语速特征...")
            zero_crossings = librosa.feature.zero_crossing_rate(y)[0]
            tempo = np.mean(zero_crossings)
            
            # 4. 梅尔频谱特征
            self.logger.info("正在提取梅尔频谱特征...")
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_mean = np.mean(mel_spec)
            
            features = {
                'pitch': float(pitch_mean),
                'energy': float(energy_mean),
                'energy_std': float(energy_std),
                'tempo': float(tempo),
                'mel_mean': float(mel_mean)
            }
            
            self.logger.info("音频特征提取完成")
            self.logger.debug(f"提取的特征: {features}")
            
            return features
            
        except Exception as e:
            self.logger.error(f"音频特征提取失败: {str(e)}")
            raise
    
    def analyze_text_emotion(self, text):
        """
        分析文本情感
        
        参数:
            text (str): 待分析的文本
            
        返回:
            dict: 情感分析结果
        """
        try:
            self.logger.info("开始文本情感分析...")
            self.logger.debug(f"待分析文本: {text[:100]}...")  # 只记录前100个字符
            
            # 对文本进行情感分析
            result = self.text_analyzer(text)
            
            # 获取情感标签和得分
            label = result[0]['label']
            score = result[0]['score']
            
            # 根据得分确定情感
            if label == 'positive':
                sentiment = '积极'
            elif label == 'negative':
                sentiment = '消极'
            else:
                # 如果模型返回其他标签，根据置信度判断
                if score > 0.6:
                    sentiment = '积极'
                elif score < 0.4:
                    sentiment = '消极'
                else:
                    sentiment = '平静'
            
            analysis_result = {
                'sentiment': sentiment,
                'confidence': float(score)
            }
            
            self.logger.info(f"文本情感分析完成: {sentiment} (置信度: {score:.2f})")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"文本情感分析失败: {str(e)}")
            # 发生异常时返回默认的平静情感
            return {
                'sentiment': '平静',
                'confidence': 0.5
            }
    
    def analyze_emotion(self, audio_path, text):
        """
        综合分析音频和文本的情感
        
        参数:
            audio_path (str): 音频文件路径
            text (str): 转录文本
            
        返回:
            dict: 综合情感分析结果
        """
        try:
            self.logger.info("开始综合情感分析...")
            
            # 获取音频特征
            audio_features = self.analyze_audio_features(audio_path)
            
            # 获取文本情感
            text_emotion = self.analyze_text_emotion(text)
            
            # 综合分析结果
            overall_emotion = self._determine_overall_emotion(audio_features, text_emotion)
            
            result = {
                'audio_features': audio_features,
                'text_emotion': text_emotion,
                'overall_emotion': overall_emotion
            }
            
            self.logger.info("综合情感分析完成")
            self.logger.info(f"整体情感: {overall_emotion['primary_emotion']} (强度: {overall_emotion['intensity']})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"综合情感分析失败: {str(e)}")
            raise
    
    def _determine_overall_emotion(self, audio_features, text_emotion):
        """
        根据音频特征和文本情感确定整体情感
        
        参数:
            audio_features (dict): 音频特征
            text_emotion (dict): 文本情感分析结果
            
        返回:
            dict: 整体情感判断结果
        """
        try:
            self.logger.info("开始确定整体情感...")
            
            # 基于规则的情感判断
            energy_threshold = 0.1
            pitch_threshold = 200
            
            # 情感强度判断
            intensity = 'normal'
            if audio_features['energy'] > energy_threshold:
                intensity = 'strong'
            
            # 获取基础情感
            base_sentiment = text_emotion['sentiment']
            confidence = text_emotion['confidence']
            
            # 如果置信度过低，根据音频特征调整基础情感
            if confidence < 0.6:
                if audio_features['energy'] > energy_threshold and audio_features['pitch'] > pitch_threshold:
                    base_sentiment = '积极'
                elif audio_features['energy'] < energy_threshold and audio_features['pitch'] < pitch_threshold:
                    base_sentiment = '平静'
                else:
                    base_sentiment = '消极'
            
            # 根据音频特征确定详细情感
            detailed_emotions = []
            
            # 根据能量和音高判断详细情感
            if base_sentiment == '积极':
                if audio_features['energy'] > energy_threshold:
                    detailed_emotions.append('热情')
                if audio_features['pitch'] > pitch_threshold:
                    detailed_emotions.append('愉快')
                if audio_features['tempo'] > 0.5:
                    detailed_emotions.append('专注')
                    
            elif base_sentiment == '消极':
                if audio_features['energy'] < energy_threshold:
                    detailed_emotions.append('疲惫')
                if audio_features['pitch'] > pitch_threshold:
                    detailed_emotions.append('紧张')
                if audio_features['tempo'] > 0.5:
                    detailed_emotions.append('焦虑')
                    
            else:  # 平静
                if audio_features['energy'] < energy_threshold:
                    detailed_emotions.append('自然')
                if audio_features['pitch'] < pitch_threshold:
                    detailed_emotions.append('放松')
                if audio_features['tempo'] < 0.5:
                    detailed_emotions.append('沉稳')
            
            # 确保至少有一个详细情感
            if not detailed_emotions:
                detailed_emotions = [self.detailed_emotions[base_sentiment][0]]
            
            result = {
                'primary_emotion': base_sentiment,
                'detailed_emotions': detailed_emotions,
                'intensity': intensity,
                'confidence': confidence
            }
            
            self.logger.info(f"整体情感确定完成: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"确定整体情感失败: {str(e)}")
            raise 