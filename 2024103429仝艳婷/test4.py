import pygame
import random
import time
import math
import os
import sys

# 初始化pygame
pygame.init()

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
DARK_BLUE = (0, 0, 139)
LIGHT_BLUE = (173, 216, 230)

# 游戏设置
BLOCK_SIZE = 25
GRID_WIDTH = 25
GRID_HEIGHT = 30
GAME_AREA_LEFT = 0  # 游戏区域左对齐
SIDEBAR_WIDTH = BLOCK_SIZE * 8  # 侧边栏宽度
SCREEN_WIDTH = BLOCK_SIZE * GRID_WIDTH + SIDEBAR_WIDTH  # 总宽度
SCREEN_HEIGHT = BLOCK_SIZE * GRID_HEIGHT

# 背景渐变颜色
BACKGROUND_COLORS = [
    (25, 25, 112),  # 午夜蓝
    (0, 0, 139),  # 深蓝
    (0, 0, 205),  # 中蓝
    (65, 105, 225),  # 皇家蓝
    (100, 149, 237)  # 矢车菊蓝
]

# 方块形状
SHAPES = [
    [[1, 1, 1, 1]],  # I
    [[1, 1], [1, 1]],  # O
    [[1, 1, 1], [0, 1, 0]],  # T
    [[1, 1, 1], [1, 0, 0]],  # L
    [[1, 1, 1], [0, 0, 1]],  # J
    [[0, 1, 1], [1, 1, 0]],  # S
    [[1, 1, 0], [0, 1, 1]]  # Z
]

# 方块颜色
COLORS = [CYAN, YELLOW, MAGENTA, ORANGE, BLUE, GREEN, RED]

# 创建游戏窗口
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("俄罗斯方块")


# 字体设置
def load_font():
    font_path = None
    possible_fonts = [
        "msyh.ttc",  # 微软雅黑
        "simhei.ttf",  # 黑体
        "simsun.ttc",  # 宋体
        "Arial Unicode.ttf",
        "NotoSansCJK-Regular.ttc"
    ]

    # 检查系统字体目录
    font_dirs = []
    if sys.platform == "win32":
        # Windows系统可能的字体目录
        font_dirs.append(os.path.join(os.environ.get('SYSTEMROOT', 'C:\\Windows'), 'Fonts'))
        font_dirs.append(os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts'))
    elif sys.platform == "darwin":
        # Mac系统字体目录
        font_dirs.append("/Library/Fonts")
        font_dirs.append(os.path.expanduser("~/Library/Fonts"))
    else:  # linux
        # Linux系统字体目录
        font_dirs.append("/usr/share/fonts")
        font_dirs.append(os.path.expanduser("~/.fonts"))

    # 先在当前目录查找，然后在系统字体目录查找
    search_paths = ['.'] + font_dirs

    for font_file in possible_fonts:
        for path in search_paths:
            full_path = os.path.join(path, font_file)
            if os.path.exists(full_path):
                font_path = full_path
                break
        if font_path:
            break

    try:
        if font_path:
            main_font = pygame.font.Font(font_path, 24)
            small_font = pygame.font.Font(font_path, 18)
            title_font = pygame.font.Font(font_path, 48)
            button_font = pygame.font.Font(font_path, 32)
            print(f"成功加载字体: {font_path}")
        else:
            print("未找到中文字体文件，使用系统默认字体")
            main_font = pygame.font.SysFont("Arial", 24)
            small_font = pygame.font.SysFont("Arial", 18)
            title_font = pygame.font.SysFont("Arial", 48)
            button_font = pygame.font.SysFont("Arial", 32)
    except Exception as e:
        print(f"字体加载失败: {e}，使用系统默认字体")
        main_font = pygame.font.SysFont(None, 24)
        small_font = pygame.font.SysFont(None, 18)
        title_font = pygame.font.SysFont(None, 48)
        button_font = pygame.font.SysFont(None, 32)

    return main_font, small_font, title_font, button_font


font, small_font, title_font, button_font = load_font()


class Button:
    def __init__(self, x, y, width, height, text, color, hover_color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.is_hovered = False

    def draw(self, surface):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=10)
        pygame.draw.rect(surface, WHITE, self.rect, 2, border_radius=10)

        text_surface = button_font.render(self.text, True, WHITE)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def check_hover(self, pos):
        self.is_hovered = self.rect.collidepoint(pos)
        return self.is_hovered

    def is_clicked(self, pos, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(pos)
        return False


class Tetromino:
    def __init__(self):
        self.shape_index = random.randint(0, len(SHAPES) - 1)
        self.shape = SHAPES[self.shape_index]
        self.color = COLORS[self.shape_index]
        self.x = GRID_WIDTH // 2 - len(self.shape[0]) // 2
        self.y = 0
        self.rotation = 0
        self.rotation_angle = 0  # 当前旋转角度(0-360)
        self.target_rotation = 0  # 目标旋转角度
        self.rotating = False  # 是否正在旋转
        self.rotation_speed = 10  # 旋转速度(度/帧)

    def rotate(self, clockwise=True):
        """旋转方块并保持中心位置准确"""
        if clockwise:
            self.target_rotation = (self.target_rotation + 90) % 360
        else:
            self.target_rotation = (self.target_rotation - 90) % 360
        self.rotating = True

        # 计算旋转后的形状
        rows = len(self.shape)
        cols = len(self.shape[0])
        rotated = [[0 for _ in range(rows)] for _ in range(cols)]

        for r in range(rows):
            for c in range(cols):
                if clockwise:
                    rotated[c][rows - 1 - r] = self.shape[r][c]
                else:
                    rotated[cols - 1 - c][r] = self.shape[r][c]

        # 调整位置以保持中心点不变
        if len(self.shape) != len(rotated) or len(self.shape[0]) != len(rotated[0]):
            # 形状尺寸变化时调整位置
            self.x += (len(self.shape[0]) - len(rotated[0])) // 2
            self.y += (len(self.shape) - len(rotated)) // 2

        return rotated

    def update_rotation(self):
        if self.rotating:
            angle_diff = (self.target_rotation - self.rotation_angle) % 360
            if angle_diff > 180:
                angle_diff -= 360

            if abs(angle_diff) <= self.rotation_speed:
                self.rotation_angle = self.target_rotation
                self.rotating = False
            else:
                self.rotation_angle = (self.rotation_angle +
                                       math.copysign(self.rotation_speed, angle_diff)) % 360

    def draw(self, surface, x_offset=0, y_offset=0, cell_size=None):
        if cell_size is None:
            cell_size = BLOCK_SIZE

        self.update_rotation()

        center_x = (self.x + len(self.shape[0]) / 2) * cell_size
        center_y = (self.y + len(self.shape) / 2) * cell_size

        for y, row in enumerate(self.shape):
            for x, cell in enumerate(row):
                if cell:
                    orig_x = (self.x + x + x_offset) * cell_size
                    orig_y = (self.y + y + y_offset) * cell_size

                    if self.rotation_angle != 0:
                        rel_x = orig_x - center_x
                        rel_y = orig_y - center_y
                        angle_rad = math.radians(self.rotation_angle)
                        new_x = rel_x * math.cos(angle_rad) - rel_y * math.sin(angle_rad)
                        new_y = rel_x * math.sin(angle_rad) + rel_y * math.cos(angle_rad)
                        orig_x = new_x + center_x
                        orig_y = new_y + center_y

                    rect = pygame.Rect(
                        orig_x,
                        orig_y,
                        cell_size - 1, cell_size - 1
                    )
                    pygame.draw.rect(surface, self.color, rect)
                    pygame.draw.rect(surface, WHITE, rect, 1)


class TetrisGame:
    def __init__(self):
        self.grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.current_piece = Tetromino()
        self.next_piece = Tetromino()
        self.game_over = False
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.fall_speed = 0.5
        self.last_fall_time = time.time()
        self.paused = False
        self.ghost_piece = None
        self.bg_color_index = 0
        self.bg_color_time = 0
        self.level_up_lines = 10
        self.level_speeds = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05]
        self.update_ghost_piece()
        self.current_level = 1
        self.max_level = 5  # 设置最大关卡数
        self.level_targets = [300, 800, 1500, 3000, 5000]  # 每个关卡的目标分数
        self.level_completed = False
        self.level_start_time = time.time()
        self.level_time_limit = 120  # 每个关卡的时间限制(秒)
        self.in_menu = True  # 是否在菜单界面
        self.start_button = Button(
            SCREEN_WIDTH // 2 - 100,
            SCREEN_HEIGHT // 2 + 50,
            200, 60,
            "开始游戏",
            (0, 100, 200),
            (0, 150, 255)
        )

    def reset(self):
        self.grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.current_piece = Tetromino()
        self.next_piece = Tetromino()
        self.game_over = False
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.fall_speed = self.level_speeds[0]
        self.last_fall_time = time.time()
        self.paused = False
        self.update_ghost_piece()
        self.level_completed = False
        self.level_start_time = time.time()

    def update_background_color(self):
        current_time = time.time()
        if current_time - self.bg_color_time > 5:
            self.bg_color_index = (self.bg_color_index + 1) % len(BACKGROUND_COLORS)
            self.bg_color_time = current_time

    def get_background_color(self):
        return BACKGROUND_COLORS[self.bg_color_index]

    def update_ghost_piece(self):
        if self.current_piece:
            # 创建新的幽灵方块实例，而不是修改现有实例
            self.ghost_piece = Tetromino()

            # 复制所有属性（包括旋转状态）
            for attr in ['shape_index', 'shape', 'x', 'y', 'rotation',
                         'rotation_angle', 'target_rotation', 'rotating', 'rotation_speed']:
                setattr(self.ghost_piece, attr, getattr(self.current_piece, attr))

            # 设置半透明颜色
            self.ghost_piece.color = (150, 150, 150, 100)

            # 重置y位置从顶部开始下落
            self.ghost_piece.y = 0

            # 下落到底部
            while self.valid_position(self.ghost_piece, 0, 1):
                self.ghost_piece.y += 1

    def valid_position(self, piece, x_offset=0, y_offset=0):
        for y, row in enumerate(piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    new_x = piece.x + x + x_offset
                    new_y = piece.y + y + y_offset

                    if (new_x < 0 or new_x >= GRID_WIDTH or
                            new_y >= GRID_HEIGHT or
                            (new_y >= 0 and self.grid[new_y][new_x])):
                        return False
        return True

    def merge_piece(self):
        for y, row in enumerate(self.current_piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    grid_y = self.current_piece.y + y
                    grid_x = self.current_piece.x + x
                    if grid_y >= 0:
                        self.grid[grid_y][grid_x] = self.current_piece.color

    def clear_lines(self):
        lines_to_clear = []
        for y in range(GRID_HEIGHT):
            if all(self.grid[y]):
                lines_to_clear.append(y)

        for line in lines_to_clear:
            del self.grid[line]
            self.grid.insert(0, [0 for _ in range(GRID_WIDTH)])

        if len(lines_to_clear) > 0:
            self.lines_cleared += len(lines_to_clear)
            self.score += self.calculate_score(len(lines_to_clear))
            new_level = self.lines_cleared // self.level_up_lines + 1
            if new_level > self.level and new_level <= len(self.level_speeds):
                self.level = new_level
                self.fall_speed = self.level_speeds[self.level - 1]

    def calculate_score(self, lines):
        base_scores = {1: 100, 2: 300, 3: 500, 4: 800}
        return base_scores.get(lines, 0) * self.level

    def move_left(self):
        if not self.paused and not self.game_over:
            if self.valid_position(self.current_piece, -1):
                self.current_piece.x -= 1
                self.update_ghost_piece()

    def move_right(self):
        if not self.paused and not self.game_over:
            if self.valid_position(self.current_piece, 1):
                self.current_piece.x += 1
                self.update_ghost_piece()

    def move_down(self):
        if not self.paused and not self.game_over:
            if self.valid_position(self.current_piece, 0, 1):
                self.current_piece.y += 1
                self.update_ghost_piece()
                return True
            else:
                self.merge_piece()
                self.clear_lines()
                self.current_piece = self.next_piece
                self.next_piece = Tetromino()
                self.update_ghost_piece()
                if not self.valid_position(self.current_piece):
                    self.game_over = True
                return False
        return True

    def hard_drop(self):
        if not self.paused and not self.game_over:
            while self.move_down():
                pass

    def rotate_piece(self, clockwise=True):
        if not self.paused and not self.game_over:
            rotated_shape = self.current_piece.rotate(clockwise)
            original_shape = self.current_piece.shape
            self.current_piece.shape = rotated_shape

            if not self.valid_position(self.current_piece):
                for offset in [1, -1, 2, -2]:
                    self.current_piece.x += offset
                    if self.valid_position(self.current_piece):
                        break
                    self.current_piece.x -= offset
                else:
                    self.current_piece.shape = original_shape

            # 完全重置幽灵方块
            self.update_ghost_piece()

    def update(self):
        current_time = time.time()
        if (current_time - self.last_fall_time > self.fall_speed and
                not self.paused and not self.game_over and not self.level_completed):
            self.move_down()
            self.last_fall_time = current_time

        self.update_background_color()
        # 检查是否完成当前关卡
        if not self.level_completed and self.score >= self.level_targets[self.current_level - 1]:
            self.level_completed = True

        # 检查时间限制
        if current_time - self.level_start_time > self.level_time_limit and not self.level_completed:
            self.game_over = True

    def draw_menu(self, surface):
        # 绘制渐变背景
        bg_color = self.get_background_color()
        surface.fill(bg_color)

        # 绘制游戏标题
        title_text = title_font.render("俄罗斯方块", True, YELLOW)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT //4))
        surface.blit(title_text, title_rect)

        # 绘制游戏说明
        instructions = [
            "游戏规则:",
            "1. 消除行数获得分数",
            "2. 每关有目标分数",
            "3. 在时间限制内完成目标",
            "4. 完成所有5个关卡获胜"
        ]

        for i, line in enumerate(instructions):
            text = small_font.render(line, True, WHITE)
            text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3 + i * 30))
            surface.blit(text, text_rect)

        # 绘制开始按钮
        self.start_button.draw(surface)

        # 绘制控制说明
        controls = [
            "控制:",
            "← → : 左右移动",
            "↑ : 顺时针旋转",
            "↓ : 加速下落",
            "空格 : 硬降",
            "Z : 逆时针旋转",
            "P : 暂停",
            "R : 重置"
        ]

        for i, line in enumerate(controls):
            text = small_font.render(line, True, WHITE)
            text_rect = text.get_rect(topleft=(50, SCREEN_HEIGHT - 180 + i * 25))
            surface.blit(text, text_rect)

    def draw_game(self, surface):
        bg_color = self.get_background_color()
        surface.fill(bg_color)

        # 绘制游戏区域背景
        pygame.draw.rect(
            surface, (30, 30, 60),
            (GAME_AREA_LEFT, 0, BLOCK_SIZE * GRID_WIDTH, BLOCK_SIZE * GRID_HEIGHT)
        )

        # 绘制网格线
        for x in range(GRID_WIDTH + 1):
            pygame.draw.line(
                surface, (50, 50, 100),
                (GAME_AREA_LEFT + x * BLOCK_SIZE, 0),
                (GAME_AREA_LEFT + x * BLOCK_SIZE, GRID_HEIGHT * BLOCK_SIZE)
            )
        for y in range(GRID_HEIGHT + 1):
            pygame.draw.line(
                surface, (50, 50, 100),
                (GAME_AREA_LEFT, y * BLOCK_SIZE),
                (GAME_AREA_LEFT + GRID_WIDTH * BLOCK_SIZE, y * BLOCK_SIZE)
            )

        # 绘制已落下的方块
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.grid[y][x]:
                    rect = pygame.Rect(
                        GAME_AREA_LEFT + x * BLOCK_SIZE,
                        y * BLOCK_SIZE,
                        BLOCK_SIZE - 1, BLOCK_SIZE - 1
                    )
                    pygame.draw.rect(surface, self.grid[y][x], rect)
                    pygame.draw.rect(surface, WHITE, rect, 1)

        # 绘制幽灵方块
        if self.ghost_piece:
            # 确保使用当前方块的旋转状态
            self.ghost_piece.rotation_angle = self.current_piece.rotation_angle
            self.ghost_piece.target_rotation = self.current_piece.target_rotation
            self.ghost_piece.rotating = self.current_piece.rotating

            ghost_color = (*self.current_piece.color[:3], 80)
            center_x = (self.ghost_piece.x + len(self.ghost_piece.shape[0]) / 2) * BLOCK_SIZE
            center_y = (self.ghost_piece.y + len(self.ghost_piece.shape) / 2) * BLOCK_SIZE

            # 强制同步旋转角度（解决旋转动画不同步的问题）
            self.ghost_piece.rotation_angle = self.current_piece.rotation_angle
            self.ghost_piece.target_rotation = self.current_piece.target_rotation

            for y, row in enumerate(self.ghost_piece.shape):
                for x, cell in enumerate(row):
                    if cell:
                        orig_x = (self.ghost_piece.x + x) * BLOCK_SIZE
                        orig_y = (self.ghost_piece.y + y) * BLOCK_SIZE

                        if self.ghost_piece.rotation_angle != 0:
                            rel_x = orig_x - center_x
                            rel_y = orig_y - center_y
                            angle_rad = math.radians(self.ghost_piece.rotation_angle)
                            orig_x = rel_x * math.cos(angle_rad) - rel_y * math.sin(angle_rad) + center_x
                            orig_y = rel_x * math.sin(angle_rad) + rel_y * math.cos(angle_rad) + center_y

                        orig_x = max(GAME_AREA_LEFT, min(orig_x, GAME_AREA_LEFT + GRID_WIDTH * BLOCK_SIZE - BLOCK_SIZE))
                        orig_y = max(0, min(orig_y, GRID_HEIGHT * BLOCK_SIZE - BLOCK_SIZE))

                        rect = pygame.Rect(
                            orig_x,
                            orig_y,
                            BLOCK_SIZE - 1, BLOCK_SIZE - 1
                        )
                        s = pygame.Surface((BLOCK_SIZE - 1, BLOCK_SIZE - 1), pygame.SRCALPHA)
                        s.fill(ghost_color)
                        surface.blit(s, (orig_x, orig_y))
                        pygame.draw.rect(surface, WHITE, rect, 1)

        # 绘制当前方块
        self.current_piece.draw(surface, GAME_AREA_LEFT // BLOCK_SIZE)

        # 绘制侧边栏
        sidebar_x = GRID_WIDTH * BLOCK_SIZE + 10

        # 绘制侧边栏背景
        pygame.draw.rect(
            surface, (40, 40, 80),
            (sidebar_x - 10, 0, SCREEN_WIDTH - sidebar_x + 10, SCREEN_HEIGHT)
        )

        # 绘制下一个方块预览
        next_text = font.render("下一个:", True, WHITE)
        surface.blit(next_text, (sidebar_x, 20))

        preview_x = sidebar_x + BLOCK_SIZE
        preview_y = 70
        preview_size = BLOCK_SIZE * 0.7

        pygame.draw.rect(
            surface, (30, 30, 60),
            (preview_x - 5, preview_y - 5,
             preview_size * 4 + 10, preview_size * 4 + 10)
        )

        for y, row in enumerate(self.next_piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    rect = pygame.Rect(
                        preview_x + x * preview_size,
                        preview_y + y * preview_size,
                        preview_size - 1, preview_size - 1
                    )
                    pygame.draw.rect(surface, self.next_piece.color, rect)
                    pygame.draw.rect(surface, WHITE, rect, 1)

        # 绘制分数信息
        score_text = font.render(f"分数: {self.score}", True, WHITE)
        surface.blit(score_text, (sidebar_x, 150))

        level_text = font.render(f"等级: {self.level}", True, WHITE)
        surface.blit(level_text, (sidebar_x, 190))

        lines_text = font.render(f"行数: {self.lines_cleared}", True, WHITE)
        surface.blit(lines_text, (sidebar_x, 230))

        speed_text = font.render(f"速度: {1 / self.fall_speed:.1f}", True, WHITE)
        surface.blit(speed_text, (sidebar_x, 270))

        # 绘制控制说明
        controls = [
            "控制:",
            "← → : 左右移动",
            "↑ : 顺时针旋转",
            "↓ : 加速下落",
            "空格 : 硬降",
            "Z : 逆时针旋转",
            "P : 暂停",
            "R : 重置"
        ]
        info_y_positions = {
            'next': 20,
            'score': 150,
            'level': 190,
            'lines': 230,
            'speed': 270,
            'stage': 310,
            'target': 340,
            'time': 370,
            'controls': 420  # 控制说明下移
        }
        # 绘制关卡信息
        level_info = font.render(f"关卡: {self.current_level}/{self.max_level}", True, WHITE)
        surface.blit(level_info, (sidebar_x, info_y_positions['stage']))

        # 绘制目标分数
        target_text = small_font.render(f"目标: {self.level_targets[self.current_level - 1]}", True, WHITE)
        surface.blit(target_text, (sidebar_x, info_y_positions['target']))

        # 绘制剩余时间
        time_left = max(0, self.level_time_limit - (time.time() - self.level_start_time))
        time_text = small_font.render(f"剩余时间: {int(time_left)}秒", True, WHITE)
        surface.blit(time_text, (sidebar_x, info_y_positions['time']))

        for i, line in enumerate(controls):
            control_text = small_font.render(line, True, WHITE)
            surface.blit(control_text, (sidebar_x, info_y_positions['controls'] + i * 25))

        # 绘制游戏暂停/结束信息
        if self.paused:
            s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            surface.blit(s, (0, 0))

            pause_text = font.render("游戏暂停", True, YELLOW)
            text_rect = pause_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            surface.blit(pause_text, text_rect)

            resume_text = small_font.render("按P继续游戏", True, WHITE)
            resume_rect = resume_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 40))
            surface.blit(resume_text, resume_rect)

        elif self.game_over:
            s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            surface.blit(s, (0, 0))

            if self.level_completed:
                # 胜利界面
                game_over_text = font.render("关卡完成!", True, GREEN)
                text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 40))
                surface.blit(game_over_text, text_rect)

                if self.current_level < self.max_level:
                    next_text = small_font.render(f"准备进入第 {self.current_level + 1} 关", True, WHITE)
                    next_rect = next_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
                    surface.blit(next_text, next_rect)

                    continue_text = small_font.render("按N进入下一关", True, WHITE)
                    continue_rect = continue_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 40))
                    surface.blit(continue_text, continue_rect)
                else:
                    congrats_text = small_font.render("恭喜通关所有关卡!", True, YELLOW)
                    congrats_rect = congrats_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
                    surface.blit(congrats_text, congrats_rect)

                    final_score = small_font.render(f"最终分数: {self.score}", True, WHITE)
                    score_rect = final_score.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 40))
                    surface.blit(final_score, score_rect)
            else:
                # 失败界面
                game_over_text = font.render("很遗憾，未能完成关卡", True, RED)
                text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 40))
                surface.blit(game_over_text, text_rect)

                target_text = small_font.render(f"目标分数: {self.level_targets[self.current_level - 1]}", True, WHITE)
                target_rect = target_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
                surface.blit(target_text, target_rect)

                score_text = small_font.render(f"你的分数: {self.score}", True, WHITE)
                score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 40))
                surface.blit(score_text, score_rect)

                restart_text = small_font.render("按R重新尝试本关", True, WHITE)
                restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 80))
                surface.blit(restart_text, restart_rect)

    def draw(self, surface):
        if self.in_menu:
            self.draw_menu(surface)
        else:
            self.draw_game(surface)


def main():
    clock = pygame.time.Clock()
    game = TetrisGame()
    running = True

    while running:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if game.in_menu:
                    if event.key == pygame.K_RETURN:  # 按回车键开始游戏
                        game.in_menu = False
                else:  # 游戏界面
                    if event.key == pygame.K_LEFT:
                        game.move_left()
                    elif event.key == pygame.K_RIGHT:
                        game.move_right()
                    elif event.key == pygame.K_DOWN:
                        game.move_down()
                    elif event.key == pygame.K_UP:
                        game.rotate_piece(clockwise=True)
                    elif event.key == pygame.K_z:
                        game.rotate_piece(clockwise=False)
                    elif event.key == pygame.K_SPACE:
                        game.hard_drop()
                    elif event.key == pygame.K_p:
                        game.paused = not game.paused
                    elif event.key == pygame.K_r:
                        game.reset()
                    elif event.key == pygame.K_n and game.level_completed and game.current_level < game.max_level:
                        game.current_level += 1
                        game.reset()
                        game.level_completed = False
            elif event.type == pygame.MOUSEMOTION:
                if game.in_menu:
                    game.start_button.check_hover(mouse_pos)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if game.in_menu and game.start_button.is_clicked(mouse_pos, event):
                    game.in_menu = False

        if not game.in_menu and not game.paused and not game.game_over:
            game.update()

        game.draw(screen)
        pygame.display.flip()

        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()