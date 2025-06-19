# 智能课堂分析系统

## 项目概述

智能课堂分析系统是一个基于多模态分析的教育技术平台，能够对课堂视频进行深度分析，提取关键信息，并提供教学洞察。系统支持视频上传、自动转录、行为识别、多模态分析和智能问答等功能。

## 系统核心功能

- **视频处理与转录**：支持H.265/HEVC格式视频上传，自动转录课堂对话
- **行为识别**：基于改进的YOLO模型，识别超过20种课堂行为
- **多模态对齐**：使用CLIP模型实现视觉-文本跨模态关联
- **智能问答**：基于DeepSeek大语言模型，针对课堂内容提供深度分析和洞察
- **可视化分析**：提供丰富的可视化工具，展示课堂行为分布、互动模式等

## 技术架构

系统采用分层架构设计：

1. **数据采集层**：视频上传、预处理、多模态传感器数据融合
2. **算法处理层**：行为识别、多模态对齐、语音处理
3. **大模型应用层**：DeepSeek模型集成、提示工程、多租户支持
4. **可视化与存储层**：3D热力图、时序分析、分布式存储

## 关键技术

- 行为识别：基于YOLOv8的实时检测模型
- 多模态对齐：CLIP-ViT-L/14模型
- 语音处理：Whisper-Large-v3
- 大语言模型：DeepSeek-MoE-16B
- 可视化：Echarts-GL
- 存储：MinIO分布式对象存储

## 行为分类原则

系统采用弗兰德斯互动分析系统(Flanders Interaction Analysis Categories)作为行为分类的理论基础，对课堂互动行为进行系统化分类和分析。

## 项目结构

```
智能课堂分析系统/
├── app/                    # 应用主目录
│   ├── api/                # API接口
│   ├── core/               # 核心功能
│   ├── models/             # 数据模型
│   │   └── schemas.py      # 数据架构定义
│   ├── services/           # 服务模块
│   │   ├── video_processor.py    # 视频处理服务
│   │   ├── behavior_detector.py  # 行为识别服务
│   │   └── multimodal_aligner.py # 多模态对齐服务
│   ├── utils/              # 工具函数
│   ├── static/             # 静态资源
│   └── templates/          # 模板文件
├── config/                 # 配置文件
│   └── settings.py         # 系统配置
├── data/                   # 数据目录
│   ├── uploads/            # 上传文件
│   └── processed/          # 处理后文件
├── .env.example            # 环境变量示例
└── requirements.txt        # 依赖库
```

## 安装与使用

### 环境要求

- Python 3.8+
- CUDA 11.7+ (可选，用于GPU加速)
- FFmpeg 4.4+

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/yourusername/classroom-analytics.git
cd classroom-analytics
```

2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 配置环境变量
```bash
cp .env.example .env
# 编辑.env文件，设置必要的环境变量
```

5. 下载预训练模型
```bash
# 创建模型目录
mkdir -p models
# 下载YOLOv8模型
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt -O models/yolov8x.pt
```

### 运行系统

```bash
# 启动API服务
python -m app.api.main
```

## 开发进度

- [x] 系统架构设计
- [x] 数据模型定义
- [x] 视频处理服务
- [x] 行为识别服务
- [x] 多模态对齐服务
- [ ] 语音转录服务
- [ ] 大语言模型集成
- [ ] API接口实现
- [ ] Web前端开发
- [ ] 可视化模块
- [ ] 系统测试与优化 