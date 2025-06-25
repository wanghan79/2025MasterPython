# 使用Whisper模型将音频转录为文本
import whisper
import os
import json
from datetime import datetime
import warnings
import logging
import shutil
import re
from pathlib import Path
import numpy as np
import jieba

# 禁用特定的 FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning, module='whisper')

# 设置日志级别为 ERROR，只显示错误信息
logging.getLogger('whisper').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

class Transcriber:
    """
    音频转录器类
    使用OpenAI的Whisper模型将音频转录为文本
    """
    
    def __init__(self, model_name="large-v2"):  # 改用 large-v2 模型，避免 v3 的重复问题
        """
        初始化转录器
        """
        self.model_name = model_name
        self.model = self._load_model_with_retry()
        
    def _load_model_with_retry(self, max_retries=3):
        """尝试加载模型，如果失败则重试"""
        whisper_cache = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
        
        for attempt in range(max_retries):
            try:
                logger.info(f"尝试加载Whisper模型 (第{attempt + 1}次尝试)")
                
                if attempt > 0:
                    logger.info("清除现有的模型缓存...")
                    if os.path.exists(whisper_cache):
                        shutil.rmtree(whisper_cache)
                        logger.info("模型缓存已清除")
                
                model = whisper.load_model(self.model_name)
                logger.info("Whisper模型加载成功")
                return model
                
            except (RuntimeError, Exception) as e:
                logger.error(f"模型加载失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"无法加载Whisper模型，已尝试{max_retries}次: {str(e)}")
                logger.info("将在下次尝试前清除缓存并重新下载...")
        
    def transcribe(self, audio_path, output_dir):
        """
        转录音频文件并保存结果
        """
        try:
            # 使用更优化的参数进行转录
            result = self.model.transcribe(
                audio_path,
                language="zh",
                initial_prompt="This is a clear Chinese teaching video. Please ignore background noise and music. Only transcribe human voice with clear, coherent content without repetition. 这是一段清晰的中文教学视频，请忽略背景噪音和音乐，只转录人声部分，内容清晰连贯，没有重复。重点是没有重复",
                no_speech_threshold=0.85,       # 提高无声音检测阈值，避免空白处产生幻觉
                logprob_threshold=-0.4,         # 进一步提高概率阈值
                compression_ratio_threshold=1.35,# 进一步降低压缩比阈值
                condition_on_previous_text=False,# 禁用上下文关联，避免错误传递
                temperature=0.0,                # 使用确定性采样
                best_of=3,                      # 生成多个结果并选择最佳
                beam_size=3,                    # 使用适中的束搜索
                patience=2.0,                   # 提高搜索耐心值
                suppress_tokens=[-1],           # 抑制特殊标记
                word_timestamps=True,           # 启用词级时间戳
                vad_filter=True,               # 启用语音活动检测
                vad_parameters={
                    "min_silence_duration_ms": 500,  # 最小静音持续时间
                    "speech_pad_ms": 400,           # 语音片段填充
                    "threshold": 0.45               # 语音检测阈值
                }
            )
            
            # 对转录结果进行后处理
            cleaned_text = self._post_process_text(result["text"])
            result["text"] = cleaned_text
            
            # 处理分段结果
            segments = self._filter_segments(result["segments"])  # 先过滤无效片段
            segments = self._merge_short_segments(segments)      # 再合并短片段
            segments = self._clean_segments(segments)           # 最后清理文本
            
            # 准备输出数据结构
            transcript_data = {
                "text": cleaned_text,
                "segments": segments,
                "created_at": datetime.now().isoformat()
            }
            
            # 保存结果
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_file = os.path.join(output_dir, f"{base_name}_transcript.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, ensure_ascii=False, indent=2)
                
            return output_file
            
        except Exception as e:
            logger.error(f"转录失败: {str(e)}")
            raise
            
    def _post_process_text(self, text):
        """
        对转录文本进行后处理，智能处理重复内容
        """
        if not text:
            return text
            
        # 1. 基本清理
        text = text.strip()
        
        # 2. 移除连续的标点符号
        text = re.sub(r'[，。！？；：、]{2,}', lambda m: m.group()[0], text)
        
        # 3. 智能分句处理
        sentences = self._split_sentences(text)
        cleaned_sentences = []
        
        # 记录已处理的句子及其出现次数
        sentence_count = {}
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            # 检查是否是有意义的重复
            if self._is_meaningful_repetition(sentence, cleaned_sentences, sentence_count):
                cleaned_sentences.append(sentence)
                sentence_count[sentence] = sentence_count.get(sentence, 0) + 1
        
        text = '。'.join(cleaned_sentences)
        if text and not text.endswith(('。', '！', '？')):
            text += '。'
        
        # 4. 修复常见的转录错误
        text = self._fix_common_errors(text)
        
        return text

    def _split_sentences(self, text):
        """
        智能分句，处理可能的句子边界
        """
        # 使用更简单的分句规则
        sentence_ends = r"([。！？])+|(\n{2,})"
        sentences = re.split(sentence_ends, text)
        # 过滤None和空字符串
        sentences = [s for s in sentences if s and s.strip()]
        return sentences

    def _is_meaningful_repetition(self, sentence, previous_sentences, sentence_count):
        """
        判断重复是否有意义
        """
        if not sentence:
            return False
            
        # 1. 如果是第一次出现，直接接受
        if sentence not in sentence_count:
            return True
            
        # 2. 分析句子的语义特征
        words = jieba.lcut(sentence)
        
        # 检查是否包含表示重复或强调的关键词
        emphasis_words = {'再', '又', '还', '重复', '继续', '反复', '多次', '一遍又一遍'}
        has_emphasis = any(word in emphasis_words for word in words)
        
        # 3. 检查重复次数
        current_count = sentence_count.get(sentence, 0)
        
        # 4. 检查上下文
        if previous_sentences:
            last_sentence = previous_sentences[-1]
            
            # 如果上一句完全不同，且当前句已重复过多次，可能是错误
            if not self._is_similar(sentence, last_sentence) and current_count >= 2:
                return False
            
            # 如果包含表示重复的关键词，允许适度重复
            if has_emphasis and current_count < 3:
                return True
            
            # 检查是否是对话场景（包含引号或对话标记）
            is_dialogue = '"' in sentence or '"' in sentence or '：' in sentence
            if is_dialogue and current_count < 3:
                return True
        
        # 5. 分析句子长度和复杂度
        if len(words) > 5 and current_count < 2:  # 较长的句子允许重复一次
            return True
            
        # 6. 默认情况：只允许重复一次
        return current_count < 1

    def _remove_internal_duplicates(self, text):
        """
        智能处理句子内部的重复内容
        """
        if not text:
            return text
            
        # 1. 分词并标注词性
        words = jieba.lcut(text)
        
        # 2. 智能处理重复
        cleaned_words = []
        i = 0
        while i < len(words):
            current_word = words[i]
            
            # 检查是否是合理的重复
            if i < len(words) - 1 and current_word == words[i + 1]:
                # 允许重复的情况：
                # 1. 表示强调的词（"很很"、"非常非常"等）
                # 2. 象声词（"哈哈"、"嘿嘿"等）
                # 3. 重叠词（"渐渐"、"轻轻"等）
                if len(current_word) == 1 or self._is_valid_repetition(current_word):
                    cleaned_words.extend([current_word] * 2)
                    i += 2
                else:
                    cleaned_words.append(current_word)
                    i += 1
            else:
                cleaned_words.append(current_word)
                i += 1
        
        return ''.join(cleaned_words)

    def _is_valid_repetition(self, word):
        """
        判断词语重复是否合理
        """
        # 1. 常见的合理重复词列表
        valid_repeats = {
            '很', '非常', '特别', '最', '太',  # 程度词
            '哈', '呵', '嘿', '嘻',           # 象声词
            '渐', '轻', '慢', '快',           # 描述词
            '一', '看', '想', '说'            # 其他常见重复
        }
        
        # 2. 检查是否在合理重复列表中
        if word in valid_repeats:
            return True
            
        # 3. 检查是否是叠词（AA型）
        if len(word) == 1:
            return True
            
        return False

    def _fix_common_errors(self, text):
        """
        智能修复转录错误
        """
        # 1. 基础错误修正
        corrections = {
            '计算计算': '计算',
            '那那': '那',
            '的的': '的',
            '是是': '是',
            '和和': '和'
        }
        
        # 2. 保留合理的重复
        preserve_repeats = {
            '很很': '很很',        # 表示程度
            '非常非常': '非常非常',  # 表示程度
            '一直一直': '一直一直',  # 表示持续
            '渐渐': '渐渐',        # 固定叠词
            '常常': '常常',        # 固定叠词
            '偶偶': '偶偶',        # 固定叠词
            '多多': '多多'         # 固定叠词
        }
        
        # 3. 先处理需要保留的重复
        for repeat, keep in preserve_repeats.items():
            text = text.replace(repeat, f"<KEEP>{keep}<KEEP>")
        
        # 4. 处理错误重复
        for error, correction in corrections.items():
            text = text.replace(error, correction)
        
        # 5. 恢复被保留的重复
        text = re.sub(r'<KEEP>(.*?)<KEEP>', r'\1', text)
        
        return text
        
    def _merge_short_segments(self, segments):
        """
        合并过短的片段
        """
        if not segments:
            return segments
            
        merged = []
        current = segments[0].copy()
        
        for next_seg in segments[1:]:
            # 如果当前片段太短，或者与下一个片段时间间隔很小
            if (next_seg['start'] - current['end'] < 0.3 or
                len(current['text']) < 5):
                current['text'] += next_seg['text']
                current['end'] = next_seg['end']
            else:
                merged.append(current)
                current = next_seg.copy()
                
        merged.append(current)
        return merged
        
    def _clean_segments(self, segments):
        """
        清理分段文本
        """
        cleaned_segments = []
        for segment in segments:
            cleaned_text = self._post_process_text(segment['text'])
            if cleaned_text:  # 只保留非空的分段
                segment['text'] = cleaned_text
                cleaned_segments.append(segment)
                
        return cleaned_segments

    def _filter_segments(self, segments):
        """
        过滤无效的语音片段
        """
        if not segments:
            return segments
            
        filtered = []
        for segment in segments:
            # 检查片段时长
            duration = segment['end'] - segment['start']
            # 过滤掉过短或可能是噪音的片段
            if duration >= 0.3 and len(segment['text'].strip()) > 0:
                # 检查文本是否包含过多重复
                text = segment['text']
                words = jieba.lcut(text)
                word_freq = {}
                for word in words:
                    if len(word) > 1:  # 只统计多字词的频率
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                # 如果任何词的重复次数超过3次，可能是幻觉产生的重复
                if all(freq <= 3 for freq in word_freq.values()):
                    filtered.append(segment)
                    
        return filtered 