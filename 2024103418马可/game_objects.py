# greedy_snake/game_objects.py

import pygame
import random
from config import *

class Snake:
    """管理蛇的状态、移动和绘制 (无变化)"""
    def __init__(self):
        self.reset()

    def reset(self):
        start_x = GRID_WIDTH // 2
        start_y = GRID_HEIGHT // 2
        self.body = [(start_x, start_y), (start_x - 1, start_y), (start_x - 2, start_y)]
        self.direction = RIGHT
        self.next_direction = RIGHT
        self.is_growing = False

    def get_head_position(self):
        return self.body[0]

    def turn(self, new_direction):
        if (new_direction[0] * -1, new_direction[1] * -1) == self.direction:
            return
        self.next_direction = new_direction

    def move(self):
        self.direction = self.next_direction
        head_x, head_y = self.get_head_position()
        dir_x, dir_y = self.direction
        new_head = (head_x + dir_x, head_y + dir_y)
        self.body.insert(0, new_head)
        if self.is_growing:
            self.is_growing = False
        else:
            self.body.pop()

    def grow(self):
        self.is_growing = True

    def check_collision_with_obstacles(self, obstacles):
        head = self.get_head_position()
        # 撞墙
        if (head[0] < 0 or head[0] >= GRID_WIDTH or
            head[1] < 0 or head[1] >= GRID_HEIGHT):
            return True
        # 撞自己
        if head in self.body[1:]:
            return True
        # 撞障碍物
        if head in obstacles:
            return True
        return False

    def draw(self, surface, theme):
        colors = THEMES[theme]
        for i, pos in enumerate(self.body):
            rect = pygame.Rect(pos[0] * GRID_SIZE, pos[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            if i == 0:
                pygame.draw.rect(surface, colors["snake_head"], rect)
                pygame.draw.rect(surface, colors["snake_body"], rect.inflate(-4, -4))
            else:
                pygame.draw.rect(surface, colors["snake_body"], rect)
                pygame.draw.rect(surface, colors["snake_skin"], rect.inflate(-6, -6))

class Food:
    """管理食物的状态和绘制"""
    def __init__(self):
        self.position = (0, 0)
        self.place_randomly([], []) # 初始放置

    def place_randomly(self, snake_body, obstacles):
        all_obstacles = set(snake_body) | set(obstacles)
        while True:
            self.position = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if self.position not in all_obstacles:
                break

    def draw(self, surface, theme):
        colors = THEMES[theme]
        rect = pygame.Rect(self.position[0] * GRID_SIZE, self.position[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(surface, colors["food"], rect)


class Obstacle:
    """管理障碍物"""
    def __init__(self):
        self.positions = set()

    def generate(self, snake_body):
        self.positions.clear()
        # 确保障碍物不生成在蛇的初始位置附近
        safe_zone = set()
        for i in range(-3, 4):
            for j in range(-3, 4):
                safe_zone.add((GRID_WIDTH//2 + i, GRID_HEIGHT//2 + j))

        for _ in range(OBSTACLE_COUNT):
            length = random.randint(OBSTACLE_MIN_LENGTH, OBSTACLE_MAX_LENGTH)
            # 随机选择水平或垂直方向
            is_horizontal = random.choice([True, False])
            
            # 尝试10次找到一个不与现有障碍物或安全区冲突的位置
            for _ in range(10):
                start_x = random.randint(1, GRID_WIDTH - length - 1)
                start_y = random.randint(1, GRID_HEIGHT - length - 1)
                new_obstacle = set()
                valid = True
                for i in range(length):
                    if is_horizontal:
                        pos = (start_x + i, start_y)
                    else:
                        pos = (start_x, start_y + i)
                    
                    if pos in self.positions or pos in snake_body or pos in safe_zone:
                        valid = False
                        break
                    new_obstacle.add(pos)
                
                if valid:
                    self.positions.update(new_obstacle)
                    break
    
    def get_all_positions(self):
        return self.positions

    def draw(self, surface, theme):
        colors = THEMES[theme]
        for pos in self.positions:
            rect = pygame.Rect(pos[0] * GRID_SIZE, pos[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(surface, colors["obstacle"], rect)
            pygame.draw.rect(surface, colors["grid"], rect.inflate(-4, -4)) # 增加立体感