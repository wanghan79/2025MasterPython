import cv2
import numpy as np
import argparse
import os
from datetime import datetime
from detector import ObjectDetector
from visualizer import Visualizer
from utils import load_config, setup_logger, save_results
from dataset import ImageDataset, VideoDataset
from evaluator import Evaluator


def main(args):
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    logger = setup_logger("object_detection", config['log_dir'])
    logger.info(f"目标检测程序启动，配置文件: {args.config}")
    
    # 初始化检测器
    detector = ObjectDetector(config)
    logger.info("检测器初始化完成")
    
    # 初始化可视化器
    visualizer = Visualizer(config)
    logger.info("可视化器初始化完成")
    
    # 初始化评估器
    evaluator = Evaluator(config)
    logger.info("评估器初始化完成")
    
    # 处理输入
    if args.input_type == "image":
        dataset = ImageDataset(args.input, config)
    elif args.input_type == "video":
        dataset = VideoDataset(args.input, config)
    else:
        logger.error(f"不支持的输入类型: {args.input_type}")
        return
    
    logger.info(f"开始处理 {args.input_type} 输入: {args.input}")
    
    # 创建输出目录
    output_dir = os.path.join(config['output_dir'], 
                             datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每一帧/图像
    results = []
    for idx, (image, image_path) in enumerate(dataset):
        if image is None:
            logger.warning(f"无法加载图像: {image_path}")
            continue
            
        # 目标检测
        detections = detector.detect(image)
        
        # 记录结果
        results.append({
            "image_path": image_path,
            "detections": detections
        })
        
        # 可视化
        if args.visualize:
            vis_image = visualizer.visualize(image, detections)
            output_path = os.path.join(output_dir, 
                                      f"vis_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, vis_image)
        
        # 显示进度
        if (idx + 1) % 10 == 0:
            logger.info(f"已处理 {idx + 1}/{len(dataset)}")
    
    # 保存检测结果
    if results:
        save_results(results, os.path.join(output_dir, "detection_results.json"))
        logger.info(f"检测结果已保存至: {output_dir}/detection_results.json")
        
        # 评估模型性能
        if args.evaluate and config.get('ground_truth_dir'):
            metrics = evaluator.evaluate(results)
            evaluator.save_metrics(metrics, os.path.join(output_dir, "evaluation_metrics.json"))
            logger.info(f"评估指标已保存至: {output_dir}/evaluation_metrics.json")
            evaluator.plot_metrics(metrics, output_dir)
    else:
        logger.warning("未检测到任何目标")
    
    logger.info("目标检测程序完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="目标检测程序")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--input", type=str, required=True, help="输入图像/视频路径或目录")
    parser.add_argument("--input_type", type=str, default="image", choices=["image", "video"], help="输入类型")
    parser.add_argument("--visualize", action="store_true", help="是否可视化检测结果")
    parser.add_argument("--evaluate", action="store_true", help="是否评估模型性能")
    args = parser.parse_args()
    
    main(args)    