# coding: utf-8

"""
Pepper平板高质量SVG表情图像库
适用于Python 2.7
"""

import sys
import time
import argparse
import traceback
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SVGFace')

# 处理编码问题
reload(sys)
sys.setdefaultencoding('utf-8')

# 导入NAOqi API
try:
    from naoqi import ALProxy
except ImportError:
    logger.error(u"错误: 无法导入NAOqi模块")
    logger.error(u"请确保已安装pynaoqi SDK并正确设置环境变量")
    sys.exit(1)

# 高质量SVG表情图像库 - 只保留脸部表情
SVG_EMOJIS = {
    # 笑脸表情 - 更圆润、更生动
    "smile": """<svg width="200" height="200" viewBox="0 0 200 200">
        <!-- 脸部 -->
        <circle cx="100" cy="100" r="90" fill="#FFDE00" stroke="#FF9500" stroke-width="2" />
        <linearGradient id="face-grad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style="stop-color:#FFDE00;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#FFCC00;stop-opacity:1" />
        </linearGradient>
        <circle cx="100" cy="100" r="88" fill="url(#face-grad)" />
        
        <!-- 眼睛 -->
        <g>
            <ellipse cx="65" cy="80" rx="12" ry="16" fill="white" />
            <circle cx="65" cy="80" r="8" fill="#333333" />
            <circle cx="68" cy="77" r="3" fill="white" />
        </g>
        <g>
            <ellipse cx="135" cy="80" rx="12" ry="16" fill="white" />
            <circle cx="135" cy="80" r="8" fill="#333333" />
            <circle cx="138" cy="77" r="3" fill="white" />
        </g>
        
        <!-- 嘴巴 -->
        <path d="M 65 130 Q 100 170 135 130" stroke="#333333" stroke-width="8" fill="none" stroke-linecap="round" />
        
        <!-- 脸颊 -->
        <circle cx="55" cy="115" r="15" fill="#FF9999" opacity="0.6" />
        <circle cx="145" cy="115" r="15" fill="#FF9999" opacity="0.6" />
    </svg>""",
    
    # 伤心脸 - 更生动的表情
    "sad": """<svg width="200" height="200" viewBox="0 0 200 200">
        <!-- 脸部 -->
        <circle cx="100" cy="100" r="90" fill="#FFDE00" stroke="#FF9500" stroke-width="2" />
        <linearGradient id="face-grad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style="stop-color:#FFDE00;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#FFCC00;stop-opacity:1" />
        </linearGradient>
        <circle cx="100" cy="100" r="88" fill="url(#face-grad)" />
        
        <!-- 眼睛 -->
        <g>
            <ellipse cx="65" cy="80" rx="12" ry="16" fill="white" />
            <circle cx="65" cy="85" r="8" fill="#333333" />
            <circle cx="67" cy="82" r="3" fill="white" />
        </g>
        <g>
            <ellipse cx="135" cy="80" rx="12" ry="16" fill="white" />
            <circle cx="135" cy="85" r="8" fill="#333333" />
            <circle cx="137" cy="82" r="3" fill="white" />
        </g>
        
        <!-- 眉毛 -->
        <path d="M 50 70 Q 65 60 80 65" stroke="#333333" stroke-width="4" fill="none" stroke-linecap="round" />
        <path d="M 150 70 Q 135 60 120 65" stroke="#333333" stroke-width="4" fill="none" stroke-linecap="round" />
        
        <!-- 嘴巴 -->
        <path d="M 65 150 Q 100 130 135 150" stroke="#333333" stroke-width="8" fill="none" stroke-linecap="round" />
        
        <!-- 泪滴 -->
        <path d="M 65 90 Q 60 100 65 110" stroke="#5599FF" stroke-width="3" fill="#88BBFF" />
        <path d="M 135 90 Q 140 100 135 110" stroke="#5599FF" stroke-width="3" fill="#88BBFF" />
    </svg>""",
    
    # 惊讶脸 - 更强调惊讶表情
    "surprise": """<svg width="200" height="200" viewBox="0 0 200 200">
        <!-- 脸部 -->
        <circle cx="100" cy="100" r="90" fill="#FFDE00" stroke="#FF9500" stroke-width="2" />
        <linearGradient id="face-grad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style="stop-color:#FFDE00;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#FFCC00;stop-opacity:1" />
        </linearGradient>
        <circle cx="100" cy="100" r="88" fill="url(#face-grad)" />
        
        <!-- 眼睛 -->
        <g>
            <ellipse cx="65" cy="80" rx="15" ry="20" fill="white" stroke="#333333" stroke-width="2" />
            <circle cx="65" cy="80" r="10" fill="#333333" />
            <circle cx="68" cy="77" r="3" fill="white" />
        </g>
        <g>
            <ellipse cx="135" cy="80" rx="15" ry="20" fill="white" stroke="#333333" stroke-width="2" />
            <circle cx="135" cy="80" r="10" fill="#333333" />
            <circle cx="138" cy="77" r="3" fill="white" />
        </g>
        
        <!-- 眉毛 -->
        <path d="M 50 55 Q 65 45 80 55" stroke="#333333" stroke-width="4" fill="none" stroke-linecap="round" />
        <path d="M 150 55 Q 135 45 120 55" stroke="#333333" stroke-width="4" fill="none" stroke-linecap="round" />
        
        <!-- 嘴巴 -->
        <ellipse cx="100" cy="140" rx="25" ry="30" fill="#333333" />
        <ellipse cx="100" cy="140" rx="20" ry="25" fill="#CC0000" />
        <ellipse cx="100" cy="150" rx="10" ry="12" fill="#330000" />
    </svg>""",
    
    # 爱心脸 - 更精致的爱心眼
    "love": """<svg width="200" height="200" viewBox="0 0 200 200">
        <!-- 脸部 -->
        <circle cx="100" cy="100" r="90" fill="#FFDE00" stroke="#FF9500" stroke-width="2" />
        <linearGradient id="face-grad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style="stop-color:#FFDE00;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#FFCC00;stop-opacity:1" />
        </linearGradient>
        <circle cx="100" cy="100" r="88" fill="url(#face-grad)" />
        
        <!-- 爱心眼睛 -->
        <g>
            <path d="M 50 70 C 50 60, 60 50, 70 55 C 80 50, 90 60, 90 70 C 90 85, 70 95, 50 70" fill="#FF5555" />
        </g>
        <g>
            <path d="M 150 70 C 150 60, 140 50, 130 55 C 120 50, 110 60, 110 70 C 110 85, 130 95, 150 70" fill="#FF5555" />
        </g>
        
        <!-- 嘴巴 -->
        <path d="M 65 130 Q 100 170 135 130" stroke="#333333" stroke-width="8" fill="none" stroke-linecap="round" />
        
        <!-- 脸颊 -->
        <circle cx="55" cy="115" r="15" fill="#FF9999" opacity="0.8" />
        <circle cx="145" cy="115" r="15" fill="#FF9999" opacity="0.8" />
        
        <!-- 飘动的爱心 -->
        <g opacity="0.8">
            <path d="M 160 40 C 160 30, 170 20, 180 25 C 190 20, 200 30, 200 40 C 200 55, 180 65, 160 40" fill="#FF5555" />
            <animateTransform attributeName="transform" type="translate" values="0,0; -5,-5; 0,0" dur="2s" repeatCount="indefinite" />
        </g>
        <g opacity="0.8">
            <path d="M 30 30 C 30 20, 40 10, 50 15 C 60 10, 70 20, 70 30 C 70 45, 50 55, 30 30" fill="#FF5555" />
            <animateTransform attributeName="transform" type="translate" values="0,0; 5,-8; 0,0" dur="3s" repeatCount="indefinite" />
        </g>
    </svg>""",
    
    # 思考脸 - 更具表现力
    "think": """<svg width="200" height="200" viewBox="0 0 200 200">
        <!-- 脸部 -->
        <circle cx="100" cy="100" r="90" fill="#FFDE00" stroke="#FF9500" stroke-width="2" />
        <linearGradient id="face-grad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style="stop-color:#FFDE00;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#FFCC00;stop-opacity:1" />
        </linearGradient>
        <circle cx="100" cy="100" r="88" fill="url(#face-grad)" />
        
        <!-- 眼睛 - 一只闭上 -->
        <g>
            <path d="M 55 80 Q 65 75 75 80" stroke="#333333" stroke-width="4" fill="none" stroke-linecap="round" />
        </g>
        <g>
            <ellipse cx="135" cy="80" rx="12" ry="16" fill="white" />
            <circle cx="135" cy="80" r="8" fill="#333333" />
            <circle cx="138" cy="77" r="3" fill="white" />
        </g>
        
        <!-- 眉毛 -->
        <path d="M 50 70 Q 60 65 70 70" stroke="#333333" stroke-width="4" fill="none" stroke-linecap="round" />
        <path d="M 130 65 Q 140 60 150 65" stroke="#333333" stroke-width="4" fill="none" stroke-linecap="round" />
        
        <!-- 嘴巴 -->
        <path d="M 80 140 Q 100 130 120 140" stroke="#333333" stroke-width="6" fill="none" stroke-linecap="round" />
        
        <!-- 思考泡泡 -->
        <g>
            <circle cx="155" cy="50" r="12" fill="white" stroke="#333333" stroke-width="2" />
            <circle cx="170" cy="35" r="8" fill="white" stroke="#333333" stroke-width="2" />
            <circle cx="180" cy="25" r="5" fill="white" stroke="#333333" stroke-width="2" />
        </g>
        
        <!-- 手势 -->
        <ellipse cx="70" cy="170" rx="15" ry="8" fill="#FFDE00" stroke="#FF9500" stroke-width="2" transform="rotate(-30 70 170)" />
        <path d="M 70 170 Q 85 155 90 140" stroke="#FF9500" stroke-width="4" fill="none" stroke-linecap="round" />
    </svg>"""
}

class PepperSVGLib:
    """Pepper表情SVG库"""
    
    def __init__(self, ip="192.168.1.119", port=9559, debug=False):
        """初始化SVG库"""
        self.ip = ip
        self.port = port
        self.connected = False
        self.debug = debug
        self.tablet_service = None
        self.tts_service = None
        
    def connect(self):
        """连接到Pepper机器人服务"""
        if self.connected:
            return True
            
        try:
            # 连接平板服务
            self.tablet_service = ALProxy("ALTabletService", self.ip, self.port)
            
            # 连接语音服务
            self.tts_service = ALProxy("ALTextToSpeech", self.ip, self.port)
            self.tts_service.setLanguage("Chinese")
            
            self.connected = True
            logger.info(u"成功连接到Pepper: %s:%s", self.ip, self.port)
            return True
            
        except Exception as e:
            logger.error(u"连接到Pepper失败: %s", e)
            if self.debug:
                logger.error(traceback.format_exc())
            return False
    
    def reset_tablet(self):
        """重置平板状态"""
        if not self.connected:
            logger.error(u"错误: 未连接到Pepper")
            return False
            
        try:
            # 隐藏当前网页
            self.tablet_service.hideWebview()
            time.sleep(0.5)
            
            # 重置平板
            self.tablet_service.resetTablet()
            time.sleep(0.5)
            
            logger.debug(u"平板已重置")
            return True
            
        except Exception as e:
            logger.error(u"重置平板出错: %s", e)
            if self.debug:
                logger.error(traceback.format_exc())
            return False
    
    def show_svg(self, svg_content, title="SVG显示"):
        """显示SVG图像"""
        if not self.connected:
            logger.error(u"错误: 未连接到Pepper")
            return False
            
        try:
            # 使用字符串拼接方式处理HTML模板和SVG内容，避免格式化问题
            html_start = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>SVG Display</title>
                <style>
                    body {
                        margin: 0;
                        padding: 0;
                        height: 100vh; 
                        width: 100vw;
                        display: flex; 
                        justify-content: center; 
                        align-items: center; 
                        background: white;
                        overflow: hidden;
                    }
                    .container {
                        width: 90vw;
                        height: 90vh;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                    }
                    svg {
                        width: 100%; 
                        height: 100%;
                        max-width: 80vmin;
                        max-height: 80vmin;
                    }
                </style>
            </head>
            <body>
                <div class="container">
            """
            
            html_end = """
                </div>
            </body>
            </html>
            """
            
            # 拼接完整的HTML
            html = html_start + svg_content + html_end
            
            # 简单的URL编码，只处理空格和特殊字符
            url = "data:text/html;charset=utf-8," + html.replace("#", "%23").replace(" ", "%20")
            
            # 先隐藏当前webview，确保新内容能刷新显示
            try:
                self.tablet_service.hideWebview()
                time.sleep(0.3)
            except:
                pass
            
            # 显示新内容
            self.tablet_service.showWebview(str(url))
            logger.info(u"已显示: %s", title)
            
            return True
            
        except Exception as e:
            logger.error(u"显示SVG失败: %s", e)
            if self.debug:
                logger.error(traceback.format_exc())
            return False
    
    def say_chinese(self, text):
        """处理中文TTS"""
        if not self.connected:
            logger.error(u"错误: 未连接到Pepper")
            return False
            
        try:
            # 处理不同类型的文本
            if isinstance(text, unicode):
                text = text.encode('utf-8')
            elif isinstance(text, str):
                text = text.decode('utf-8').encode('utf-8')
                
            # 播放语音
            self.tts_service.say(str(text))
            logger.debug(u"说话: %s", text)
            return True
            
        except Exception as e:
            logger.error(u"语音错误: %s", e)
            if self.debug:
                logger.error(traceback.format_exc())
            return False
    
    def show_emoji(self, emoji_id, title=None, speak=True):
        """显示表情"""
        if not self.connected:
            logger.error(u"错误: 未连接到Pepper")
            return False
            
        # 检查表情ID是否存在
        if emoji_id not in SVG_EMOJIS:
            logger.error(u"错误: 未知的表情ID: %s", emoji_id)
            logger.info(u"可用的表情: %s", ", ".join(SVG_EMOJIS.keys()))
            return False
            
        # 获取表情SVG
        svg_content = SVG_EMOJIS[emoji_id]
        
        # 设置标题
        emoji_titles = {
            "smile": u"笑脸",
            "sad": u"伤心脸",
            "surprise": u"惊讶脸",
            "love": u"爱心脸",
            "think": u"思考脸"
        }
        
        if title is None:
            if emoji_id in emoji_titles:
                title = emoji_titles[emoji_id]
            else:
                title = emoji_id
        
        # 显示SVG
        success = self.show_svg(svg_content, title)
        
        # 语音提示
        if success and speak:
            self.say_chinese(u"这是" + title)
            
        return success
    
    def demo_emojis(self):
        """演示所有表情"""
        if not self.connected:
            logger.error(u"错误: 未连接到Pepper")
            return False
            
        logger.info(u"==== SVG表情演示 ====")
        
        # 重置平板
        self.reset_tablet()
        
        # 语音提示开始
        self.say_chinese(u"开始SVG表情演示")
        
        try:
            # 定义表情和名称
            emoji_names = {
                "smile": u"笑脸",
                "sad": u"伤心脸",
                "think": u"思考脸",
                "surprise": u"惊讶脸",
                "love": u"爱心脸"
            }
            
            # 获取表情ID列表
            emoji_ids = list(emoji_names.keys())
            total = len(emoji_ids)
            
            # 依次显示每个表情
            for i, emoji_id in enumerate(emoji_ids):
                emoji_name = emoji_names[emoji_id]
                
                # 记录日志
                logger.info(u"显示表情 %d/%d: %s", i + 1, total, emoji_name)
                
                # 显示表情
                self.show_emoji(emoji_id, emoji_name)
                
                # 等待足够长的时间
                time.sleep(3)
            
            # 演示结束
            logger.info(u"演示完成")
            self.say_chinese(u"演示完成")
            
            return True
            
        except Exception as e:
            logger.error(u"表情演示失败: %s", e)
            if self.debug:
                logger.error(traceback.format_exc())
            return False

    def get_available_emojis(self):
        """获取可用的表情列表
        
        返回:
            list: 可用表情名称列表，如['smile', 'sad', 'surprise', 'love', 'think']
        """
        return list(SVG_EMOJIS.keys())

def get_svg_service(ip="192.168.1.119", port=9559, debug=False):
    """获取SVG服务实例
    
    这个函数用于在bridge服务中获取SVG表情服务的单例实例。
    
    参数:
        ip (str): 机器人IP地址
        port (int): NAOqi端口
        debug (bool): 是否启用调试模式
        
    返回:
        PepperSVGLib: SVG服务实例
    """
    # 如果全局中已有实例，则复用
    global _svg_service_instance
    
    if '_svg_service_instance' not in globals() or _svg_service_instance is None:
        # 创建新实例
        _svg_service_instance = PepperSVGLib(ip, port, debug)
        
    # 如果IP或端口变化，则重新创建
    if _svg_service_instance.ip != ip or _svg_service_instance.port != port:
        _svg_service_instance = PepperSVGLib(ip, port, debug)
        
    return _svg_service_instance

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description=u"Pepper平板SVG表情图像库")
    parser.add_argument("--ip", type=str, default="192.168.1.119", help=u"机器人IP地址")
    parser.add_argument("--port", type=int, default=9559, help=u"NAOqi端口")
    parser.add_argument("--debug", action="store_true", help=u"启用调试模式")
    args = parser.parse_args()
    
    # 设置日志级别
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # 显示欢迎信息
    logger.info(u"Pepper平板SVG表情图像库")
    logger.info(u"=====================")
    logger.info(u"连接到机器人: %s:%s", args.ip, args.port)
    
    # 创建SVG库实例
    svg_lib = PepperSVGLib(args.ip, args.port, args.debug)
    
    # 连接到机器人
    if not svg_lib.connect():
        logger.error(u"无法连接到机器人, 程序退出")
        sys.exit(1)
    
    # 开始演示
    svg_lib.demo_emojis()

if __name__ == "__main__":
    main()   