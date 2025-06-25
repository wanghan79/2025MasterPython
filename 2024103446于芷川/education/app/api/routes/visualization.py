"""
智能课堂分析系统 - 可视化路由
"""

import os
import json
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from app.models.schemas import VisualizationRequest, VisualizationResponse
from config.settings import PROCESSED_DIR

# 创建路由
router = APIRouter()

# 行为分析数据缓存
behavior_analysis = {}

@router.post("/generate", response_model=VisualizationResponse, summary="生成可视化")
async def generate_visualization(request: VisualizationRequest):
    """
    根据请求生成可视化数据
    """
    try:
        video_id = request.video_id
        
        # 检查视频是否存在
        behavior_file = os.path.join(PROCESSED_DIR, video_id, "analysis", "behavior_analysis.json")
        
        if not os.path.exists(behavior_file):
            raise HTTPException(status_code=404, detail=f"行为分析数据不存在，请确保视频已完成处理")
        
        # 加载行为分析数据
        if video_id not in behavior_analysis:
            with open(behavior_file, "r", encoding="utf-8") as f:
                behavior_analysis[video_id] = json.load(f)
        
        # 根据可视化类型生成数据
        visualization_data = {}
        
        if request.visualization_type == "behavior_distribution":
            visualization_data = generate_behavior_distribution(
                video_id, 
                request.time_range, 
                request.behavior_filter
            )
        
        elif request.visualization_type == "behavior_timeline":
            visualization_data = generate_behavior_timeline(
                video_id, 
                request.time_range, 
                request.behavior_filter
            )
        
        elif request.visualization_type == "behavior_heatmap":
            visualization_data = generate_behavior_heatmap(
                video_id, 
                request.time_range, 
                request.behavior_filter
            )
        
        elif request.visualization_type == "attention_curve":
            visualization_data = generate_attention_curve(
                video_id, 
                request.time_range
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"不支持的可视化类型: {request.visualization_type}")
        
        # 创建响应
        response = VisualizationResponse(
            video_id=video_id,
            visualization_type=request.visualization_type,
            data=visualization_data
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"生成可视化失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成可视化失败: {str(e)}")

@router.get("/types", summary="获取可视化类型")
async def get_visualization_types():
    """
    获取支持的可视化类型列表
    """
    return {
        "types": [
            {
                "id": "behavior_distribution",
                "name": "行为分布",
                "description": "展示不同行为类别的分布情况"
            },
            {
                "id": "behavior_timeline",
                "name": "行为时间线",
                "description": "展示行为随时间的变化"
            },
            {
                "id": "behavior_heatmap",
                "name": "行为热力图",
                "description": "展示行为在视频帧中的空间分布"
            },
            {
                "id": "attention_curve",
                "name": "注意力曲线",
                "description": "展示学生注意力随时间的变化"
            }
        ]
    }

# 辅助函数：生成行为分布可视化数据
def generate_behavior_distribution(
    video_id: str, 
    time_range: Optional[List[float]] = None, 
    behavior_filter: Optional[List[str]] = None
) -> Dict[str, Any]:
    """生成行为分布可视化数据"""
    
    # 获取行为数据
    behaviors = behavior_analysis[video_id].get("behaviors", [])
    
    # 应用时间过滤
    if time_range:
        behaviors = [b for b in behaviors if time_range[0] <= b["timestamp"] <= time_range[1]]
    
    # 应用行为过滤
    if behavior_filter:
        behaviors = [b for b in behaviors if b["label"] in behavior_filter]
    
    # 统计各类行为数量
    behavior_counts = {}
    for behavior in behaviors:
        category = behavior["category"]
        label = behavior["label"]
        
        if category not in behavior_counts:
            behavior_counts[category] = {}
        
        if label not in behavior_counts[category]:
            behavior_counts[category][label] = 0
        
        behavior_counts[category][label] += 1
    
    # 生成饼图数据
    pie_data = []
    for category, labels in behavior_counts.items():
        for label, count in labels.items():
            pie_data.append({
                "name": f"{category}-{label}",
                "value": count
            })
    
    # 生成柱状图数据
    bar_data = {
        "categories": [],
        "series": []
    }
    
    for category, labels in behavior_counts.items():
        bar_data["categories"].append(category)
        
        series_item = {
            "name": category,
            "data": []
        }
        
        for label, count in labels.items():
            series_item["data"].append({
                "name": label,
                "value": count
            })
        
        bar_data["series"].append(series_item)
    
    return {
        "pie": pie_data,
        "bar": bar_data,
        "total_behaviors": len(behaviors)
    }

# 辅助函数：生成行为时间线可视化数据
def generate_behavior_timeline(
    video_id: str, 
    time_range: Optional[List[float]] = None, 
    behavior_filter: Optional[List[str]] = None
) -> Dict[str, Any]:
    """生成行为时间线可视化数据"""
    
    # 获取行为数据
    behaviors = behavior_analysis[video_id].get("behaviors", [])
    
    # 应用时间过滤
    if time_range:
        behaviors = [b for b in behaviors if time_range[0] <= b["timestamp"] <= time_range[1]]
    
    # 应用行为过滤
    if behavior_filter:
        behaviors = [b for b in behaviors if b["label"] in behavior_filter]
    
    # 按时间排序
    behaviors.sort(key=lambda x: x["timestamp"])
    
    # 生成时间线数据
    timeline_data = []
    
    for behavior in behaviors:
        timeline_data.append({
            "timestamp": behavior["timestamp"],
            "category": behavior["category"],
            "label": behavior["label"],
            "confidence": behavior["confidence"]
        })
    
    # 生成行为频率数据（每10秒统计一次）
    frequency_data = {}
    
    # 确定时间范围
    min_time = min([b["timestamp"] for b in behaviors]) if behaviors else 0
    max_time = max([b["timestamp"] for b in behaviors]) if behaviors else 0
    
    # 创建时间桶（每10秒一个桶）
    bucket_size = 10  # 10秒
    buckets = {}
    
    for i in range(int(min_time // bucket_size), int(max_time // bucket_size) + 1):
        bucket_start = i * bucket_size
        bucket_end = (i + 1) * bucket_size
        buckets[i] = {
            "time_range": [bucket_start, bucket_end],
            "behaviors": {}
        }
    
    # 统计每个时间桶中的行为
    for behavior in behaviors:
        bucket_index = int(behavior["timestamp"] // bucket_size)
        label = behavior["label"]
        
        if bucket_index in buckets:
            if label not in buckets[bucket_index]["behaviors"]:
                buckets[bucket_index]["behaviors"][label] = 0
            
            buckets[bucket_index]["behaviors"][label] += 1
    
    # 转换为序列数据
    frequency_data = {
        "x_axis": [f"{b['time_range'][0]}-{b['time_range'][1]}" for b in buckets.values()],
        "series": []
    }
    
    # 获取所有行为标签
    all_labels = set()
    for bucket in buckets.values():
        all_labels.update(bucket["behaviors"].keys())
    
    # 为每个行为创建一个系列
    for label in all_labels:
        series_data = []
        
        for bucket in buckets.values():
            series_data.append(bucket["behaviors"].get(label, 0))
        
        frequency_data["series"].append({
            "name": label,
            "data": series_data
        })
    
    return {
        "timeline": timeline_data,
        "frequency": frequency_data
    }

# 辅助函数：生成行为热力图可视化数据
def generate_behavior_heatmap(
    video_id: str, 
    time_range: Optional[List[float]] = None, 
    behavior_filter: Optional[List[str]] = None
) -> Dict[str, Any]:
    """生成行为热力图可视化数据"""
    
    # 获取行为数据
    behaviors = behavior_analysis[video_id].get("behaviors", [])
    
    # 应用时间过滤
    if time_range:
        behaviors = [b for b in behaviors if time_range[0] <= b["timestamp"] <= time_range[1]]
    
    # 应用行为过滤
    if behavior_filter:
        behaviors = [b for b in behaviors if b["label"] in behavior_filter]
    
    # 生成热力图数据
    heatmap_data = []
    
    for behavior in behaviors:
        if "bbox" in behavior:
            bbox = behavior["bbox"]
            
            # 计算边界框中心点
            center_x = (bbox["x1"] + bbox["x2"]) / 2
            center_y = (bbox["y1"] + bbox["y2"]) / 2
            
            heatmap_data.append({
                "x": center_x,
                "y": center_y,
                "value": behavior["confidence"],
                "category": behavior["category"],
                "label": behavior["label"],
                "timestamp": behavior["timestamp"]
            })
    
    return {
        "heatmap": heatmap_data,
        "dimensions": {
            "width": 1280,  # 假设视频宽度为1280
            "height": 720   # 假设视频高度为720
        }
    }

# 辅助函数：生成注意力曲线可视化数据
def generate_attention_curve(
    video_id: str, 
    time_range: Optional[List[float]] = None
) -> Dict[str, Any]:
    """生成注意力曲线可视化数据"""
    
    # 获取行为数据
    behaviors = behavior_analysis[video_id].get("behaviors", [])
    
    # 应用时间过滤
    if time_range:
        behaviors = [b for b in behaviors if time_range[0] <= b["timestamp"] <= time_range[1]]
    
    # 只关注学生行为
    student_behaviors = [b for b in behaviors if b["category"] == "student"]
    
    # 确定时间范围
    min_time = min([b["timestamp"] for b in student_behaviors]) if student_behaviors else 0
    max_time = max([b["timestamp"] for b in student_behaviors]) if student_behaviors else 0
    
    # 创建时间桶（每5秒一个桶）
    bucket_size = 5  # 5秒
    buckets = {}
    
    for i in range(int(min_time // bucket_size), int(max_time // bucket_size) + 1):
        bucket_start = i * bucket_size
        bucket_end = (i + 1) * bucket_size
        buckets[i] = {
            "time_range": [bucket_start, bucket_end],
            "attention_behaviors": 0,
            "distraction_behaviors": 0,
            "total_behaviors": 0
        }
    
    # 定义注意力和分心行为
    attention_behaviors = ["举手", "回答", "提问", "做笔记"]
    distraction_behaviors = ["低头", "转头"]
    
    # 统计每个时间桶中的行为
    for behavior in student_behaviors:
        bucket_index = int(behavior["timestamp"] // bucket_size)
        label = behavior["label"]
        
        if bucket_index in buckets:
            buckets[bucket_index]["total_behaviors"] += 1
            
            if label in attention_behaviors:
                buckets[bucket_index]["attention_behaviors"] += 1
            
            if label in distraction_behaviors:
                buckets[bucket_index]["distraction_behaviors"] += 1
    
    # 计算注意力得分
    attention_scores = []
    timestamps = []
    
    for bucket_index, bucket in buckets.items():
        if bucket["total_behaviors"] > 0:
            # 注意力得分 = (注意力行为 - 分心行为) / 总行为
            score = (bucket["attention_behaviors"] - bucket["distraction_behaviors"]) / bucket["total_behaviors"]
            # 归一化到0-100
            score = (score + 1) * 50
            
            attention_scores.append(score)
            timestamps.append(bucket["time_range"][0])
    
    return {
        "timestamps": timestamps,
        "attention_scores": attention_scores,
        "average_score": sum(attention_scores) / len(attention_scores) if attention_scores else 0
    } 