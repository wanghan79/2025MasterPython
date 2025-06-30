# greedy_snake/config.py

import pygame

# --- 屏幕与网格配置 ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRID_SIZE = 20
GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE

# --- 游戏速度 ---
FPS_NORMAL = 15
FPS_AI = 60

# --- 移动方向向量 ---
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# --- 文件路径 ---
HIGH_SCORE_FILE = "highscore.txt"

# --- 字体 ---
FONT_PRIMARY = 'simhei'
FONT_FALLBACK = 'arial'

# --- 颜色定义 (用作备用) ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# --- 游戏主题配置 ---
THEMES = {
    "经典": {
        "background": (0, 0, 0),
        "grid": (40, 40, 40),
        "snake_head": (0, 155, 0),
        "snake_body": (0, 255, 0),
        "snake_skin": (40, 40, 40),
        "food": (255, 0, 0),
        "obstacle": (128, 128, 128),
        "text": (230, 230, 230),
        "game_over": (255, 50, 50),
        "high_score": (255, 255, 0),
        "hud_ai": (0, 200, 255),
        "menu_bg": (20, 20, 60),
        "menu_title": (255, 255, 255),
        "menu_text": (200, 200, 255),
        "menu_selected": (255, 255, 0),
    },
    "森林": {
        "background": (34, 139, 34), # ForestGreen
        "grid": (60, 179, 113), # MediumSeaGreen
        "snake_head": (210, 105, 30), # Chocolate
        "snake_body": (244, 164, 96), # SandyBrown
        "snake_skin": (139, 69, 19), # SaddleBrown
        "food": (255, 69, 0), # OrangeRed (苹果)
        "obstacle": (139, 69, 19), # SaddleBrown (树桩)
        "text": (255, 255, 240), # Ivory
        "game_over": (178, 34, 34), # Firebrick
        "high_score": (255, 215, 0), # Gold
        "hud_ai": (72, 209, 204), # MediumTurquoise
        "menu_bg": (0, 100, 0), # DarkGreen
        "menu_title": (255, 248, 220), # Cornsilk
        "menu_text": (245, 222, 179), # Wheat
        "menu_selected": (255, 255, 0), # Yellow
    },
    "夜间": {
        "background": (25, 25, 112), # MidnightBlue
        "grid": (72, 61, 139), # DarkSlateBlue
        "snake_head": (255, 0, 255), # Magenta
        "snake_body": (148, 0, 211), # DarkViolet
        "snake_skin": (75, 0, 130), # Indigo
        "food": (0, 255, 255), # Cyan (能量块)
        "obstacle": (119, 136, 153), # LightSlateGray
        "text": (240, 248, 255), # AliceBlue
        "game_over": (220, 20, 60), # Crimson
        "high_score": (50, 205, 50), # LimeGreen
        "hud_ai": (30, 144, 255), # DodgerBlue
        "menu_bg": (10, 10, 40),
        "menu_title": (255, 255, 255),
        "menu_text": (176, 196, 222), # LightSteelBlue
        "menu_selected": (0, 255, 255), # Cyan
    },
}

# --- 障碍物配置 ---
OBSTACLE_COUNT = 10 # 障碍物模式下的障碍物数量
OBSTACLE_MIN_LENGTH = 3
OBSTACLE_MAX_LENGTH = 6

# --- 音效配置 ---
SOUNDS = {
    "eat": "assets/eat.wav",
    "game_over": "assets/game_over.wav",
    "click": "assets/click.wav",
}