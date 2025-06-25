"""
智能课堂分析系统 - API主入口
"""

import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger

from config.settings import API_CONFIG, BASE_DIR
from app.api.routes import videos, analysis, visualization

# 创建FastAPI应用
app = FastAPI(
    title="智能课堂分析系统",
    description="基于多模态分析的教育技术平台，能够对课堂视频进行深度分析，提取关键信息，并提供教学洞察。",
    version="0.1.0",
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "app", "static")), name="static")

# 配置模板
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "app", "templates"))

# 包含路由
app.include_router(videos.router, prefix="/api/videos", tags=["视频管理"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["视频分析"])
app.include_router(visualization.router, prefix="/api/visualization", tags=["数据可视化"])

# 首页路由
@app.get("/", tags=["页面"])
async def index(request: Request):
    """返回首页"""
    return templates.TemplateResponse("index.html", {"request": request})

# 视频详情页路由
@app.get("/videos/{video_id}", tags=["页面"])
async def video_detail(request: Request, video_id: str):
    """返回视频详情页"""
    return templates.TemplateResponse("video_detail.html", {"request": request, "video_id": video_id})

# 分析页面路由
@app.get("/analysis/{video_id}", tags=["页面"])
async def analysis_page(request: Request, video_id: str):
    """返回分析页面"""
    return templates.TemplateResponse("analysis.html", {"request": request, "video_id": video_id})

# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    logger.info("智能课堂分析系统API服务启动")

# 关闭事件
@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行"""
    logger.info("智能课堂分析系统API服务关闭")

# 运行应用
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.api.main:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["debug"],
        workers=API_CONFIG["workers"],
    ) 