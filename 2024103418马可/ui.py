# greedy_snake/ui.py

import pygame
from config import *

class UI:
    """管理所有UI元素的绘制"""
    def __init__(self, surface):
        self.surface = surface
        try:
            self.font_large = pygame.font.SysFont(FONT_PRIMARY, 60)
            self.font_medium = pygame.font.SysFont(FONT_PRIMARY, 40)
            self.font_small = pygame.font.SysFont(FONT_PRIMARY, 24)
            self.font_tiny = pygame.font.SysFont(FONT_PRIMARY, 18)
        except pygame.error:
            self.font_large = pygame.font.SysFont(FONT_FALLBACK, 60)
            self.font_medium = pygame.font.SysFont(FONT_FALLBACK, 40)
            self.font_small = pygame.font.SysFont(FONT_FALLBACK, 24)
            self.font_tiny = pygame.font.SysFont(FONT_FALLBACK, 18)

    def _draw_text(self, text, font, color, center_pos):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=center_pos)
        self.surface.blit(text_surface, text_rect)

    def draw_grid(self, theme):
        colors = THEMES[theme]
        for x in range(0, SCREEN_WIDTH, GRID_SIZE):
            pygame.draw.line(self.surface, colors["grid"], (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
            pygame.draw.line(self.surface, colors["grid"], (0, y), (SCREEN_WIDTH, y))

    def draw_hud(self, score, high_score, is_ai_mode, theme):
        colors = THEMES[theme]
        self._draw_text(f"分数: {score}", self.font_small, colors["text"], (80, 25))
        self._draw_text(f"最高分: {high_score}", self.font_small, colors["high_score"], (SCREEN_WIDTH - 120, 25))
        if is_ai_mode:
            self._draw_text("AI 模式", self.font_small, colors["hud_ai"], (SCREEN_WIDTH // 2, 25))

    def draw_start_screen(self, theme, selected_option, selected_theme_index, theme_names):
        colors = THEMES[theme]
        self.surface.fill(colors["menu_bg"])
        
        self._draw_text("贪 吃 蛇", self.font_large, colors["menu_title"], (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4))

        options = ["开始游戏", "AI 模式", "障碍物模式", "切换主题", "退出"]
        for i, option in enumerate(options):
            color = colors["menu_selected"] if i == selected_option else colors["menu_text"]
            self._draw_text(option, self.font_medium, color, (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + i * 50 - 50))

        # 显示当前主题
        theme_text = f"< {theme_names[selected_theme_index]} >"
        theme_color = colors["menu_selected"] if selected_option == 3 else colors["menu_text"]
        self._draw_text(theme_text, self.font_small, theme_color, (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 3 * 50))


    def draw_game_over_screen(self, score, high_score, theme):
        colors = THEMES[theme]
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.surface.blit(overlay, (0, 0))

        self._draw_text("游戏结束", self.font_large, colors["game_over"], (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3))
        
        score_text = f"本次得分: {score}"
        text_color = colors["text"]
        # if score > self.high_score and score > 0:
        if score > high_score and score > 0:
            score_text = f"新纪录! {score}"
            text_color = colors["high_score"]

        self._draw_text(score_text, self.font_medium, text_color, (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self._draw_text("按 [回车键] 返回主菜单", self.font_small, colors["text"], (SCREEN_WIDTH // 2, SCREEN_HEIGHT * 0.7))

    def draw_pause_screen(self, theme):
        colors = THEMES[theme]
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.surface.blit(overlay, (0, 0))
        self._draw_text("已暂停", self.font_large, colors["text"], (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20))
        self._draw_text("按 [P] 键继续", self.font_small, colors["text"], (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 40))