# greedy_snake/game_engine.py

import pygame
import sys
from config import *
from game_objects import Snake, Food, Obstacle
from ui import UI
from ai import find_path_bfs

class SoundManager:
    """管理所有音效的加载和播放"""
    def __init__(self):
        self.sounds = {}
        for name, path in SOUNDS.items():
            try:
                self.sounds[name] = pygame.mixer.Sound(path)
            except pygame.error:
                print(f"警告: 无法加载音效文件 '{path}'")
                self.sounds[name] = None
    
    def play(self, name):
        if self.sounds.get(name):
            self.sounds[name].play()

class GameEngine:
    def __init__(self):
        pygame.init()
        pygame.mixer.init() # 初始化混音器
        pygame.display.set_caption("增强版贪吃蛇")
        
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.snake = Snake()
        self.food = Food()
        self.obstacles = Obstacle()
        self.ui = UI(self.screen)
        self.sound_manager = SoundManager()
        
        self.score = 0
        self.high_score = self.load_high_score()
        
        self.game_state = 'START_MENU' # 'START_MENU', 'PLAYING', 'PAUSED', 'GAME_OVER'
        self.is_ai_mode = False
        self.obstacle_mode = False
        self.ai_path = []
        
        # 主题和菜单状态
        self.theme_names = list(THEMES.keys())
        self.selected_theme_index = 0
        self.current_theme = self.theme_names[self.selected_theme_index]
        self.menu_selected_option = 0

    def load_high_score(self):
        try:
            with open(HIGH_SCORE_FILE, 'r') as f: return int(f.read())
        except (FileNotFoundError, ValueError): return 0

    def save_high_score(self):
        with open(HIGH_SCORE_FILE, 'w') as f: f.write(str(self.high_score))

    def reset_game(self, ai_mode, obstacle_mode):
        self.is_ai_mode = ai_mode
        self.obstacle_mode = obstacle_mode
        self.snake.reset()
        
        if self.obstacle_mode:
            self.obstacles.generate(self.snake.body)
        else:
            self.obstacles.positions.clear()
            
        self.food.place_randomly(self.snake.body, self.obstacles.get_all_positions())
        self.score = 0
        self.ai_path = []
        self.game_state = 'PLAYING'
        
    def run(self):
        while True:
            self.handle_events()
            self.update()
            self.draw()
            fps = FPS_AI if self.is_ai_mode else FPS_NORMAL
            self.clock.tick(fps)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_game()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.quit_game()
                
                # 根据不同游戏状态处理按键
                if self.game_state == 'START_MENU':
                    self.handle_menu_input(event.key)
                elif self.game_state == 'PLAYING':
                    self.handle_playing_input(event.key)
                elif self.game_state == 'PAUSED':
                    if event.key == pygame.K_p:
                        self.game_state = 'PLAYING'
                elif self.game_state == 'GAME_OVER':
                    if event.key == pygame.K_RETURN:
                        self.game_state = 'START_MENU'

    def handle_menu_input(self, key):
        if key == pygame.K_UP:
            self.menu_selected_option = (self.menu_selected_option - 1) % 5
            self.sound_manager.play('click')
        elif key == pygame.K_DOWN:
            self.menu_selected_option = (self.menu_selected_option + 1) % 5
            self.sound_manager.play('click')
        elif key == pygame.K_LEFT and self.menu_selected_option == 3: # 切换主题
            self.selected_theme_index = (self.selected_theme_index - 1) % len(self.theme_names)
            self.current_theme = self.theme_names[self.selected_theme_index]
            self.sound_manager.play('click')
        elif key == pygame.K_RIGHT and self.menu_selected_option == 3: # 切换主题
            self.selected_theme_index = (self.selected_theme_index + 1) % len(self.theme_names)
            self.current_theme = self.theme_names[self.selected_theme_index]
            self.sound_manager.play('click')
        elif key == pygame.K_RETURN:
            self.sound_manager.play('click')
            if self.menu_selected_option == 0: # 开始游戏
                self.reset_game(ai_mode=False, obstacle_mode=False)
            elif self.menu_selected_option == 1: # AI 模式
                self.reset_game(ai_mode=True, obstacle_mode=False)
            elif self.menu_selected_option == 2: # 障碍物模式
                self.reset_game(ai_mode=False, obstacle_mode=True)
            elif self.menu_selected_option == 4: # 退出
                self.quit_game()

    def handle_playing_input(self, key):
        if key == pygame.K_p:
            self.game_state = 'PAUSED'
            return
            
        if not self.is_ai_mode:
            if key == pygame.K_UP: self.snake.turn(UP)
            elif key == pygame.K_DOWN: self.snake.turn(DOWN)
            elif key == pygame.K_LEFT: self.snake.turn(LEFT)
            elif key == pygame.K_RIGHT: self.snake.turn(RIGHT)
            
    def update(self):
        if self.game_state != 'PLAYING': return
            
        if self.is_ai_mode: self.update_ai()
        self.snake.move()

        if self.snake.get_head_position() == self.food.position:
            self.sound_manager.play('eat')
            self.snake.grow()
            self.score += 10
            if self.score > self.high_score: self.high_score = self.score
            self.food.place_randomly(self.snake.body, self.obstacles.get_all_positions())
            if self.is_ai_mode: self.ai_path = []

        if self.snake.check_collision_with_obstacles(self.obstacles.get_all_positions()):
            self.sound_manager.play('game_over')
            self.save_high_score()
            self.game_state = 'GAME_OVER'

    def update_ai(self):
        if not self.ai_path:
            self.ai_path = find_path_bfs(
                self.snake.get_head_position(),
                self.food.position,
                self.snake.body,
                self.obstacles.get_all_positions()
            )
            if not self.ai_path: # 备用策略
                for direction in [UP, DOWN, LEFT, RIGHT]:
                    next_pos_tuple = (self.snake.get_head_position()[0] + direction[0], self.snake.get_head_position()[1] + direction[1])
                    if not self.snake.check_collision_with_obstacles(self.obstacles.get_all_positions()):
                         self.snake.turn(direction)
                         return
        
        if self.ai_path: self.snake.turn(self.ai_path.pop(0))

    def draw(self):
        colors = THEMES[self.current_theme]
        self.screen.fill(colors["background"])
        
        if self.game_state == 'START_MENU':
            self.ui.draw_start_screen(self.current_theme, self.menu_selected_option, self.selected_theme_index, self.theme_names)
        elif self.game_state in ['PLAYING', 'PAUSED', 'GAME_OVER']:
            self.ui.draw_grid(self.current_theme)
            self.snake.draw(self.screen, self.current_theme)
            self.food.draw(self.screen, self.current_theme)
            if self.obstacle_mode: self.obstacles.draw(self.screen, self.current_theme)
            self.ui.draw_hud(self.score, self.high_score, self.is_ai_mode, self.current_theme)

            if self.game_state == 'PAUSED': self.ui.draw_pause_screen(self.current_theme)
            elif self.game_state == 'GAME_OVER': self.ui.draw_game_over_screen(self.score, self.high_score, self.current_theme)
            
        pygame.display.flip()
        
    def quit_game(self):
        pygame.quit()
        sys.exit()