"""
智能课堂分析系统 - 视频分析路由
"""

import os
import json
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Body
from loguru import logger

from app.models.schemas import AnalysisQuery, AnalysisResult, BehaviorAnalysis, Transcript
from config.settings import PROCESSED_DIR

# 创建路由
router = APIRouter()

# 分析结果存储（实际应用中应使用数据库）
analysis_results = {}
behavior_analysis = {}
transcripts = {}

@router.post("/query", response_model=AnalysisResult, summary="分析查询")
async def analyze_query(query: AnalysisQuery):
    """
    对视频内容进行智能分析查询
    """
    try:
        video_id = query.video_id
        
        # 检查视频是否存在
        behavior_file = os.path.join(PROCESSED_DIR, video_id, "analysis", "behavior_analysis.json")
        transcript_file = os.path.join(PROCESSED_DIR, video_id, "transcripts", "transcript.json")
        
        if not os.path.exists(behavior_file) or not os.path.exists(transcript_file):
            raise HTTPException(status_code=404, detail=f"视频分析数据不存在，请确保视频已完成处理")
        
        # 加载行为分析和转录数据
        if video_id not in behavior_analysis:
            with open(behavior_file, "r", encoding="utf-8") as f:
                behavior_analysis[video_id] = json.load(f)
        
        if video_id not in transcripts:
            with open(transcript_file, "r", encoding="utf-8") as f:
                transcripts[video_id] = json.load(f)
        
        # 构建分析结果
        # 注意：这里应该调用DeepSeek大语言模型进行实际分析
        # 这里只是一个示例实现
        
        # 生成一个简单的分析响应
        response = f"分析查询: {query.query}\n"
        response += f"基于视频ID: {video_id}\n"
        
        if query.time_range:
            response += f"时间范围: {query.time_range[0]}-{query.time_range[1]}秒\n"
        
        if query.behavior_filter:
            response += f"行为过滤: {', '.join(query.behavior_filter)}\n"
        
        # 添加一些行为统计信息
        if video_id in behavior_analysis:
            summary = behavior_analysis[video_id].get("summary", {})
            response += f"\n行为统计:\n"
            response += f"总帧数: {summary.get('total_frames', 0)}\n"
            response += f"总行为数: {summary.get('total_behaviors', 0)}\n"
            
            # 添加行为百分比
            percentages = summary.get("behavior_percentages", {})
            for category, behaviors in percentages.items():
                response += f"\n{category}类行为:\n"
                for behavior, percent in behaviors.items():
                    response += f"- {behavior}: {percent:.2f}%\n"
        
        # 添加一些转录信息
        if video_id in transcripts:
            segments = transcripts[video_id].get("segments", [])
            if segments:
                response += f"\n转录片段示例:\n"
                for i, segment in enumerate(segments[:3]):  # 只显示前3个片段
                    response += f"[{segment['start_time']:.1f}-{segment['end_time']:.1f}] {segment['text']}\n"
        
        # 创建分析结果
        result = AnalysisResult(
            query=query.query,
            response=response,
            video_id=video_id,
            relevant_segments=[],  # 这里应该包含相关片段
            visualization_data={}  # 这里应该包含可视化数据
        )
        
        # 存储结果
        if video_id not in analysis_results:
            analysis_results[video_id] = []
        analysis_results[video_id].append(result.dict())
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"分析查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"分析查询失败: {str(e)}")

@router.get("/{video_id}/behaviors", response_model=BehaviorAnalysis, summary="获取行为分析")
async def get_behavior_analysis(
    video_id: str,
    time_start: Optional[float] = Query(None, description="开始时间(秒)"),
    time_end: Optional[float] = Query(None, description="结束时间(秒)"),
    categories: Optional[List[str]] = Query(None, description="行为类别过滤"),
    labels: Optional[List[str]] = Query(None, description="行为标签过滤"),
):
    """
    获取视频的行为分析结果
    """
    try:
        # 检查视频是否存在
        behavior_file = os.path.join(PROCESSED_DIR, video_id, "analysis", "behavior_analysis.json")
        
        if not os.path.exists(behavior_file):
            raise HTTPException(status_code=404, detail=f"行为分析数据不存在，请确保视频已完成处理")
        
        # 加载行为分析数据
        if video_id not in behavior_analysis:
            with open(behavior_file, "r", encoding="utf-8") as f:
                behavior_analysis[video_id] = json.load(f)
        
        # 获取行为数据
        behaviors = behavior_analysis[video_id].get("behaviors", [])
        
        # 应用过滤
        filtered_behaviors = []
        for behavior in behaviors:
            # 时间过滤
            if time_start is not None and behavior["timestamp"] < time_start:
                continue
            if time_end is not None and behavior["timestamp"] > time_end:
                continue
            
            # 类别过滤
            if categories and behavior["category"] not in categories:
                continue
            
            # 标签过滤
            if labels and behavior["label"] not in labels:
                continue
            
            filtered_behaviors.append(behavior)
        
        # 创建结果
        result = {
            "video_id": video_id,
            "behaviors": filtered_behaviors,
            "model_version": behavior_analysis[video_id].get("model_version", ""),
            "summary": behavior_analysis[video_id].get("summary", {})
        }
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取行为分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取行为分析失败: {str(e)}")

@router.get("/{video_id}/transcript", response_model=Transcript, summary="获取转录文本")
async def get_transcript(
    video_id: str,
    time_start: Optional[float] = Query(None, description="开始时间(秒)"),
    time_end: Optional[float] = Query(None, description="结束时间(秒)"),
):
    """
    获取视频的转录文本
    """
    try:
        # 检查视频是否存在
        transcript_file = os.path.join(PROCESSED_DIR, video_id, "transcripts", "transcript.json")
        
        if not os.path.exists(transcript_file):
            raise HTTPException(status_code=404, detail=f"转录数据不存在，请确保视频已完成处理")
        
        # 加载转录数据
        if video_id not in transcripts:
            with open(transcript_file, "r", encoding="utf-8") as f:
                transcripts[video_id] = json.load(f)
        
        # 获取转录片段
        segments = transcripts[video_id].get("segments", [])
        
        # 应用时间过滤
        filtered_segments = []
        for segment in segments:
            # 时间过滤
            if time_start is not None and segment["end_time"] < time_start:
                continue
            if time_end is not None and segment["start_time"] > time_end:
                continue
            
            filtered_segments.append(segment)
        
        # 创建结果
        result = {
            "video_id": video_id,
            "language": transcripts[video_id].get("language", "zh"),
            "segments": filtered_segments,
            "full_text": " ".join([segment["text"] for segment in filtered_segments]),
            "model_version": transcripts[video_id].get("model_version", "")
        }
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取转录文本失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取转录文本失败: {str(e)}")

@router.get("/{video_id}/history", response_model=List[AnalysisResult], summary="获取分析历史")
async def get_analysis_history(video_id: str):
    """
    获取视频的分析查询历史
    """
    try:
        if video_id not in analysis_results:
            return []
        
        return analysis_results[video_id]
    
    except Exception as e:
        logger.error(f"获取分析历史失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取分析历史失败: {str(e)}") 