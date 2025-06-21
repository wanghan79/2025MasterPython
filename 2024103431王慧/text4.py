import pygame
import random
import math
import sys
import os

# 初始化pygame
pygame.init()
pygame.mixer.init()

# 屏幕设置
WIDTH, HEIGHT = 1000, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("星际矿工")
clock = pygame.time.Clock()

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
BLUE = (0, 150, 255)
ORANGE = (255, 140, 0)
PURPLE = (160, 32, 240)
METAL = (180, 180, 190)
DARK_BLUE = (0, 50, 100)
MENU_BG = (20, 20, 40)

# 字体 - 使用更可靠的加载方式
try:
    font = pygame.font.SysFont('simhei', 24)  # 使用系统支持的字体
    big_font = pygame.font.SysFont('simhei', 48)
    title_font = pygame.font.SysFont('simhei', 64)
except:
    font = pygame.font.Font(None, 24)  # 回退到默认字体
    big_font = pygame.font.Font(None, 48)
    title_font = pygame.font.Font(None, 64)

# 星星背景
stars = [(random.randint(0, WIDTH), random.randint(0, HEIGHT), random.uniform(0.1, 0.5)) for _ in range(200)]


class Player:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.speed = 5
        self.angle = 0
        self.radius = 25
        self.health = 100
        self.fuel = 100
        self.max_fuel = 100
        self.minerals = {"iron": 0, "gold": 0, "crystal": 0}
        self.lasers = []
        self.laser_cooldown = 0
        self.upgrades = {"speed": 1, "laser": 1, "fuel": 1}
        self.thruster_timer = 0
        self.thruster_particles = []
        self.engine_glow = 0
        self.laser_cooldown = 0
        self.cooldown_max = 10  # 10帧冷却（60FPS下约0.16秒）
        self.laser_properties = {
            'range': 5000,  # 射程5000像素（可击中屏幕外目标）
            'width': 3,  # 基础宽度
            'speed': 3000,  # 激光速度（像素/秒）
            'duration': 0.5  # 显示持续时间(秒)
        }

    def draw(self, screen):
        # 绘制更真实的飞船
        center = (self.x, self.y)

        # 飞船主体 - 流线型设计
        body_points = [
            (self.x + math.cos(self.angle) * self.radius * 1.2,
             self.y + math.sin(self.angle) * self.radius * 1.2),
            (self.x + math.cos(self.angle + math.pi / 2) * self.radius * 0.6,
             self.y + math.sin(self.angle + math.pi / 2) * self.radius * 0.6),
            (self.x - math.cos(self.angle) * self.radius * 0.8,
             self.y - math.sin(self.angle) * self.radius * 0.8),
            (self.x + math.cos(self.angle - math.pi / 2) * self.radius * 0.6,
             self.y + math.sin(self.angle - math.pi / 2) * self.radius * 0.6)
        ]

        # 绘制金属机身
        pygame.draw.polygon(screen, METAL, body_points)
        pygame.draw.polygon(screen, DARK_BLUE, body_points, 2)

        # 驾驶舱 - 圆形玻璃罩
        cockpit_pos = (self.x + math.cos(self.angle) * self.radius * 0.5,
                       self.y + math.sin(self.angle) * self.radius * 0.5)
        pygame.draw.circle(screen, (100, 180, 255, 150), (int(cockpit_pos[0]), int(cockpit_pos[1])), 12)
        pygame.draw.circle(screen, (50, 100, 200), (int(cockpit_pos[0]), int(cockpit_pos[1])), 12, 2)

        # 机翼细节
        wing1_points = [
            (self.x + math.cos(self.angle + math.pi / 2) * self.radius * 0.4,
             self.y + math.sin(self.angle + math.pi / 2) * self.radius * 0.4),
            (self.x + math.cos(self.angle + math.pi / 2) * self.radius * 0.8,
             self.y + math.sin(self.angle + math.pi / 2) * self.radius * 0.8),
            (self.x + math.cos(self.angle + math.pi / 2 + math.pi / 4) * self.radius * 0.6,
             self.y + math.sin(self.angle + math.pi / 2 + math.pi / 4) * self.radius * 0.6)
        ]
        wing2_points = [
            (self.x + math.cos(self.angle - math.pi / 2) * self.radius * 0.4,
             self.y + math.sin(self.angle - math.pi / 2) * self.radius * 0.4),
            (self.x + math.cos(self.angle - math.pi / 2) * self.radius * 0.8,
             self.y + math.sin(self.angle - math.pi / 2) * self.radius * 0.8),
            (self.x + math.cos(self.angle - math.pi / 2 - math.pi / 4) * self.radius * 0.6,
             self.y + math.sin(self.angle - math.pi / 2 - math.pi / 4) * self.radius * 0.6)
        ]

        pygame.draw.polygon(screen, DARK_BLUE, wing1_points)
        pygame.draw.polygon(screen, DARK_BLUE, wing2_points)

        # 引擎部分
        engine_pos = (self.x - math.cos(self.angle) * self.radius * 0.7,
                      self.y - math.sin(self.angle) * self.radius * 0.7)
        pygame.draw.circle(screen, (80, 80, 90), (int(engine_pos[0]), int(engine_pos[1])), 10)

        # 引擎尾焰效果
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] and self.fuel > 0:
            self.engine_glow = min(self.engine_glow + 0.2, 1)

            flame_length = random.uniform(20, 30)
            flame_width = random.uniform(10, 15)

            inner_flame = [
                (self.x - math.cos(self.angle) * self.radius * 0.8,
                 self.y - math.sin(self.angle) * self.radius * 0.8),
                (self.x - math.cos(self.angle) * (self.radius + flame_length),
                 self.y - math.sin(self.angle) * (self.radius + flame_length)),
                (self.x - math.cos(self.angle) * (self.radius + flame_length * 0.7) + math.cos(
                    self.angle + 0.3) * flame_width,
                 self.y - math.sin(self.angle) * (self.radius + flame_length * 0.7) + math.sin(
                     self.angle + 0.3) * flame_width),
                (self.x - math.cos(self.angle) * (self.radius + flame_length * 0.3) + math.cos(
                    self.angle + 0.2) * flame_width * 0.7,
                 self.y - math.sin(self.angle) * (self.radius + flame_length * 0.3) + math.sin(
                     self.angle + 0.2) * flame_width * 0.7)
            ]

            pygame.draw.polygon(screen, YELLOW, inner_flame)
            pygame.draw.polygon(screen, ORANGE, inner_flame, 1)

            outer_flame = [
                (self.x - math.cos(self.angle) * self.radius * 0.8,
                 self.y - math.sin(self.angle) * self.radius * 0.8),
                (self.x - math.cos(self.angle) * (self.radius + flame_length * 0.8),
                 self.y - math.sin(self.angle) * (self.radius + flame_length * 0.8)),
                (self.x - math.cos(self.angle) * (self.radius + flame_length * 0.5) + math.cos(
                    self.angle - 0.3) * flame_width * 1.2,
                 self.y - math.sin(self.angle) * (self.radius + flame_length * 0.5) + math.sin(
                     self.angle - 0.3) * flame_width * 1.2),
                (self.x - math.cos(self.angle) * (self.radius + flame_length * 0.2) + math.cos(
                    self.angle - 0.2) * flame_width * 0.8,
                 self.y - math.sin(self.angle) * (self.radius + flame_length * 0.2) + math.sin(
                     self.angle - 0.2) * flame_width * 0.8)
            ]

            pygame.draw.polygon(screen, ORANGE, outer_flame)
            pygame.draw.polygon(screen, RED, outer_flame, 1)

            if random.random() < 0.4:
                self.thruster_particles.append([
                    self.x - math.cos(self.angle) * self.radius * 0.8,
                    self.y - math.sin(self.angle) * self.radius * 0.8,
                    self.angle + random.uniform(-0.5, 0.5),
                    random.uniform(1, 4),
                    random.uniform(0.5, 1.5)
                ])
        else:
            self.engine_glow = max(self.engine_glow - 0.05, 0)

        if self.engine_glow > 0:
            glow_color = (
                min(255, 100 + int(155 * self.engine_glow)),
                min(255, 50 + int(100 * self.engine_glow)),
                50
            )
            pygame.draw.circle(screen, glow_color, (int(engine_pos[0]), int(engine_pos[1])),
                               int(10 + 5 * self.engine_glow))

        for p in self.thruster_particles[:]:
            p[0] += math.cos(p[2]) * p[3]
            p[1] += math.sin(p[2]) * p[3]
            p[3] *= 0.9
            p[4] -= 0.05

            alpha = min(255, int(255 * p[4]))
            size = int(p[3] * 3)

            if p[4] <= 0 or size <= 0:
                self.thruster_particles.remove(p)
            else:
                particle_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, (255, 200, 100, alpha), (size, size), size)
                screen.blit(particle_surf, (int(p[0] - size), int(p[1] - size)))

        for laser in self.lasers:
            # 主激光束（更粗更亮）
            pygame.draw.line(screen, (0, 255, 0),
                             (laser[0], laser[1]),
                             (laser[0] + math.cos(laser[2]) * 50,
                              laser[1] + math.sin(laser[2]) * 50),
                             5 + self.upgrades["laser"])  # 根据升级增加宽度

            # 添加光晕效果
            glow_surf = pygame.Surface((100, 100), pygame.SRCALPHA)
            pygame.draw.line(glow_surf, (0, 255, 0, 80),
                             (50, 50),
                             (50 + math.cos(laser[2]) * 50,
                              50 + math.sin(laser[2]) * 50),
                             15 + self.upgrades["laser"] * 2)
            screen.blit(glow_surf, (laser[0] - 50, laser[1] - 50))

    def move(self, keys):
        if keys[pygame.K_LEFT]:
            self.angle -= 0.1
        if keys[pygame.K_RIGHT]:
            self.angle += 0.1

        move_x = math.cos(self.angle) * self.speed * self.upgrades["speed"]
        move_y = math.sin(self.angle) * self.speed * self.upgrades["speed"]

        if keys[pygame.K_UP] and self.fuel > 0:
            self.x += move_x
            self.y += move_y
            self.fuel -= 0.05
        if keys[pygame.K_DOWN]:
            self.x -= move_x * 0.5
            self.y -= move_y * 0.5

        self.x = max(self.radius, min(WIDTH - self.radius, self.x))
        self.y = max(self.radius, min(HEIGHT - self.radius, self.y))

        if self.laser_cooldown > 0:
            self.laser_cooldown -= 1

    def shoot(self):
        if self.laser_cooldown == 0:
            # 根据升级等级决定是否发射多束激光
            lasers_to_fire = 1
            if self.upgrades["laser"] >= 3:
                lasers_to_fire = 2
            if self.upgrades["laser"] >= 5:
                lasers_to_fire = 3

            for i in range(lasers_to_fire):
                # 添加小幅角度偏移（仅当多发时）
                angle_offset = 0
                if lasers_to_fire > 1:
                    angle_offset = (i - (lasers_to_fire - 1) / 2) * 0.1

                self.lasers.append([
                    self.x + math.cos(self.angle + angle_offset) * 30,
                    self.y + math.sin(self.angle + angle_offset) * 30,
                    self.angle + angle_offset
                ])

            self.laser_cooldown = 20 // self.upgrades["laser"]

    def update_lasers(self):
        for laser in self.lasers[:]:
            # 增加激光移动速度（原为15）
            laser[0] += math.cos(laser[2]) * 25
            laser[1] += math.sin(laser[2]) * 25

            # 放宽边界判定
            if (laser[0] < -100 or laser[0] > WIDTH + 100 or
                    laser[1] < -100 or laser[1] > HEIGHT + 100):
                self.lasers.remove(laser)

class Asteroid:
    def __init__(self, level=1):  # 这里添加默认参数
        self.radius = random.randint(20, 50) * (1 + level * 0.05)  # 随关卡增加大小
        self.x = random.choice([-self.radius, WIDTH + self.radius])
        self.y = random.randint(0, HEIGHT)
        self.speed = random.uniform(0.5, 1.0) * (1 + level * 0.05)  # 随关卡增加速度
        self.angle = math.atan2(HEIGHT // 2 - self.y, WIDTH // 2 - self.x)
        self.mineral_type = random.choice(["iron", "gold", "crystal"])
        self.health = self.radius
        self.rotation = 0
        self.rotation_speed = random.uniform(-0.02, 0.02)

        self.colors = {
            "iron": (150, 150, 150),
            "gold": (255, 215, 0),
            "crystal": (100, 200, 255)
        }

    def draw(self, screen):
        self.rotation += self.rotation_speed
        points = []
        for i in range(8):
            angle = self.rotation + i * math.pi / 4
            radius_var = self.radius * (0.8 + 0.2 * math.sin(angle * 3))
            points.append((self.x + math.cos(angle) * radius_var,
                           self.y + math.sin(angle) * radius_var))
        pygame.draw.polygon(screen, self.colors[self.mineral_type], points)

    def move(self):
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed
        if self.x < -100 or self.x > WIDTH + 100 or self.y < -100 or self.y > HEIGHT + 100:
            return True
        return False

    def mine(self, damage):
        self.health -= damage
        if self.health <= 0:
            return self.mineral_type, self.radius
        return None


class Game:
    def __init__(self):
        self.level = 1  # 当前关卡
        self.level_up_score = 200  # 升级到下一关需要的分数
        self.level_complete = False  # 标记是否完成当前关卡
        self.level_complete_time = 0  # 新增，用于记录通关时间
        self.player = Player()
        self.asteroids = []
        self.spawn_timer = 0
        self.score = 0
        self.game_over = False
        self.upgrade_menu = False
        self.wave = 1
        self.wave_timer = 0
        self.game_state = "menu"  # menu, playing, paused, game_over
        self.pause_timer = 0
        self.menu_selection = 0
        self.menu_options = ["开始游戏", "游戏说明", "退出游戏"]
        self.instructions = [
            "控制方式:",
            "方向键 - 移动飞船",
            "空格键 - 发射激光",
            "U键 - 打开/关闭升级菜单",
            "ESC键 - 暂停游戏/返回菜单",
            "",
            "游戏目标:",
            "开采小行星获取矿物",
            "使用矿物升级飞船",
            "生存尽可能长的时间"
            "",
            "关卡系统:",
            "每获得500分进入下一关",
            "关卡越高难度越大"
        ]

    def draw_menu(self, screen):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        screen.blit(overlay, (0, 0))

        # 标题
        title_bg = pygame.Surface((400, 80), pygame.SRCALPHA)
        title_bg.fill((0, 0, 50, 150))
        screen.blit(title_bg, (WIDTH // 2 - 200, 80))

        title = title_font.render("星际矿工", True, BLUE)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 100))

        # 菜单选项
        for i, option in enumerate(self.menu_options):
            bg_rect = pygame.Rect(WIDTH // 2 - 150, 250 + i * 70, 300, 50)
            pygame.draw.rect(screen, (30, 30, 60, 150), bg_rect)
            pygame.draw.rect(screen, (100, 100, 200), bg_rect, 2)

            color = YELLOW if i == self.menu_selection else WHITE
            text = big_font.render(option, True, color)
            screen.blit(text, (WIDTH // 2 - text.get_width() // 2, 250 + i * 70))

        # 控制提示
        controls_bg = pygame.Surface((400, 40), pygame.SRCALPHA)
        controls_bg.fill((0, 0, 30, 150))
        screen.blit(controls_bg, (WIDTH // 2 - 200, HEIGHT - 60))

        controls = font.render("使用方向键选择，回车键确认", True, WHITE)
        screen.blit(controls, (WIDTH // 2 - controls.get_width() // 2, HEIGHT - 50))

    def spawn_asteroids(self):
        self.spawn_timer += 1
        # 根据关卡调整生成速度 (关卡越高生成越快)
        spawn_rate = max(10, 60 // (self.wave * 0.5 + self.level * 0.3))
        if self.spawn_timer >= spawn_rate:
            self.asteroids.append(Asteroid(self.level))
            self.spawn_timer = 0

    def check_collisions(self):
        # 检查激光与小行星的碰撞
        for laser in self.player.lasers[:]:
            laser_removed = False
            for asteroid in self.asteroids[:]:
                # 激光起点和终点
                laser_start = (laser[0], laser[1])
                laser_end = (
                    laser[0] + math.cos(laser[2]) * 50,
                    laser[1] + math.sin(laser[2]) * 50
                )

                # 线段到圆心的最短距离
                closest_point = self.closest_point_on_segment(
                    laser_start, laser_end, (asteroid.x, asteroid.y))
                distance = math.hypot(asteroid.x - closest_point[0],
                                      asteroid.y - closest_point[1])

                if distance < asteroid.radius + 5:  # 增加5像素的容错
                    result = asteroid.mine(10 * self.player.upgrades["laser"])
                    if result:
                        mineral, amount = result
                        self.player.minerals[mineral] += amount
                        self.score += amount * (1 if mineral == "iron" else 2 if mineral == "gold" else 3)
                        self.asteroids.remove(asteroid)

                    # 标记激光需要移除
                    laser_removed = True
                    break  # 这个激光已经击中一个小行星，跳出内层循环

            # 在外部循环中移除激光，避免修改正在迭代的列表
            if laser_removed:
                self.player.lasers.remove(laser)

        # 检查玩家与小行星的碰撞
        for asteroid in self.asteroids[:]:
            # 计算玩家与小行星中心的距离
            distance = math.sqrt((self.player.x - asteroid.x) ** 2 +
                                 (self.player.y - asteroid.y) ** 2)

            # 如果距离小于两者半径之和，发生碰撞
            if distance < self.player.radius + asteroid.radius:
                self.player.health -= 10  # 每次碰撞减少10点生命值
                self.asteroids.remove(asteroid)  # 移除撞到的小行星

                # 检查是否游戏结束
                if self.player.health <= 0:
                    self.player.health = 0  # 确保生命值不为负
                    self.game_over = True
                    self.game_state = "game_over"

                # 可以在这里添加碰撞特效或声音
                # self.add_collision_effect(asteroid.x, asteroid.y)

    def closest_point_on_segment(self, seg_start, seg_end, point):
        # 计算线段上距离给定点最近的点
        seg_vec = (seg_end[0] - seg_start[0], seg_end[1] - seg_start[1])
        point_vec = (point[0] - seg_start[0], point[1] - seg_start[1])
        seg_length_squared = seg_vec[0] ** 2 + seg_vec[1] ** 2

        if seg_length_squared == 0:
            return seg_start

        t = max(0, min(1, (point_vec[0] * seg_vec[0] + point_vec[1] * seg_vec[1]) / seg_length_squared))
        return (
            seg_start[0] + t * seg_vec[0],
            seg_start[1] + t * seg_vec[1]
        )
    def update_wave(self):
        self.wave_timer += 1
        if self.wave_timer >= 1800:
            self.wave += 1
            self.wave_timer = 0
            self.player.fuel = min(self.player.max_fuel, self.player.fuel + 20)

    def check_level_up(self):
        # 检查是否应该升级到下一关
        if self.score >= self.level * self.level_up_score and not self.level_complete:
            self.level += 1
            self.level_complete = True
            self.level_complete_time = pygame.time.get_ticks()  # 记录通关时间（新增）

            # 清空当前小行星，为下一关做准备
            self.asteroids = []
            # 恢复玩家状态
            self.player.health = min(100, self.player.health + 20)  # 恢复一些生命值
            self.player.fuel = self.player.max_fuel  # 加满燃料
            # 增加难度
            self.wave = 1  # 重置波次但保持关卡难度
            return True
        return False

    def draw_ui(self, screen):
        if self.game_state == "playing":
            # 绘制生命值和燃料条
            pygame.draw.rect(screen, RED, (10, 10, self.player.health * 2, 20))
            pygame.draw.rect(screen, YELLOW, (10, 40, self.player.fuel * 2, 20))

            # 显示状态文本（生命、燃料、分数、波次、关卡）
            screen.blit(font.render(f"生命: {int(self.player.health)}", True, WHITE), (220, 10))
            screen.blit(font.render(f"燃料: {int(self.player.fuel)}", True, WHITE), (220, 40))
            screen.blit(font.render(f"分数: {int(self.score)}", True, WHITE), (WIDTH - 200, 10))
            screen.blit(font.render(f"波次: {int(self.wave)}", True, WHITE), (WIDTH - 200, 40))
            screen.blit(font.render(f"关卡: {int(self.level)}", True, WHITE), (WIDTH - 200, 70))

            # 显示矿物数量（强制转换为整数）
            mineral_text = f"铁: {int(self.player.minerals['iron'])} 金: {int(self.player.minerals['gold'])} 水晶: {int(self.player.minerals['crystal'])}"
            screen.blit(font.render(mineral_text, True, WHITE), (10, HEIGHT - 30))

            # 升级菜单
            if self.upgrade_menu:
                pygame.draw.rect(screen, (50, 50, 100), (WIDTH // 4, HEIGHT // 4, WIDTH // 2, HEIGHT // 2))
                screen.blit(big_font.render("升级菜单", True, WHITE), (WIDTH // 2 - 100, HEIGHT // 4 + 20))

                upgrades = [
                    ("速度", "speed", 50),
                    ("激光", "laser", 80),
                    ("燃料舱", "fuel", 30)
                ]

                for i, (name, key, cost) in enumerate(upgrades):
                    color = GREEN if self.player.minerals["iron"] >= cost else RED
                    # 升级菜单中的矿物数量也显示为整数
                    text = font.render(
                        f"{name}: {int(cost)}铁 (等级 {int(self.player.upgrades[key])})",
                        True,
                        color
                    )
                    screen.blit(text, (WIDTH // 4 + 50, HEIGHT // 4 + 100 + i * 50))

                screen.blit(font.render("按ESC返回游戏", True, WHITE), (WIDTH // 2 - 80, HEIGHT // 4 + 250))

            # 关卡进度显示
            current_level_score = (self.level - 1) * self.level_up_score
            next_level_score = self.level * self.level_up_score

            if self.score >= next_level_score and not self.level_complete:
                self.level_complete = True
                screen.blit(big_font.render("关卡完成!", True, GREEN),
                            (WIDTH // 2 - 100, HEIGHT // 2 - 50))
            elif self.level_complete:
                if pygame.time.get_ticks() - self.level_complete_time < 5000:  # 显示2秒
                    screen.blit(big_font.render("关卡完成!", True, GREEN),
                                (WIDTH // 2 - 100, HEIGHT // 2 - 50))
                else:
                    self.level_complete = False
            else:
                # 显示整数进度百分比
                progress = min(1.0, max(0.0, (self.score - current_level_score) / self.level_up_score))
                pygame.draw.rect(screen, BLUE, (WIDTH // 4, HEIGHT - 50, WIDTH // 2 * progress, 20))
                screen.blit(font.render(f"进度: {int(progress * 100)}%", True, WHITE),
                            (WIDTH // 2 - 60, HEIGHT - 50))
    def draw_game_over(self, screen):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))

        # 添加背景增强可见性
        bg = pygame.Surface((500, 300), pygame.SRCALPHA)
        bg.fill((30, 0, 0, 200))
        screen.blit(bg, (WIDTH // 2 - 250, HEIGHT // 2 - 150))

        screen.blit(big_font.render("游戏结束!", True, RED), (WIDTH // 2 - 100, HEIGHT // 2 - 100))
        screen.blit(big_font.render(f"最终分数: {self.score}", True, WHITE), (WIDTH // 2 - 120, HEIGHT // 2))
        screen.blit(font.render("按 R 键重新开始", True, WHITE), (WIDTH // 2 - 80, HEIGHT // 2 + 100))
        screen.blit(font.render("按 M 键返回菜单", True, WHITE), (WIDTH // 2 - 80, HEIGHT // 2 + 150))
        screen.blit(big_font.render(f"最终关卡: {self.level}", True, WHITE),
                    (WIDTH // 2 - 120, HEIGHT // 2 + 50))
    def reset_game(self):
        self.player = Player()
        self.asteroids = []
        self.spawn_timer = 0
        self.score = 0
        self.game_over = False
        self.upgrade_menu = False
        self.wave = 1
        self.wave_timer = 0

    def draw_menu(self, screen):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        screen.blit(overlay, (0, 0))

        # 标题
        title_bg = pygame.Surface((400, 80), pygame.SRCALPHA)
        title_bg.fill((0, 0, 50, 150))
        screen.blit(title_bg, (WIDTH // 2 - 200, 80))

        title = title_font.render("星际矿工", True, BLUE)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 100))

        # 菜单选项
        for i, option in enumerate(self.menu_options):
            bg_rect = pygame.Rect(WIDTH // 2 - 150, 250 + i * 70, 300, 50)
            pygame.draw.rect(screen, (30, 30, 60, 150), bg_rect)
            pygame.draw.rect(screen, (100, 100, 200), bg_rect, 2)

            color = YELLOW if i == self.menu_selection else WHITE
            text = big_font.render(option, True, color)
            screen.blit(text, (WIDTH // 2 - text.get_width() // 2, 250 + i * 70))

        # 控制提示
        controls_bg = pygame.Surface((400, 40), pygame.SRCALPHA)
        controls_bg.fill((0, 0, 30, 150))
        screen.blit(controls_bg, (WIDTH // 2 - 200, HEIGHT - 60))

        controls = font.render("使用方向键选择，回车键确认", True, WHITE)
        screen.blit(controls, (WIDTH // 2 - controls.get_width() // 2, HEIGHT - 50))

    def draw_instructions(self, screen):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        screen.blit(overlay, (0, 0))

        title = big_font.render("游戏说明", True, YELLOW)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 50))

        # 说明文本背景
        text_bg = pygame.Surface((600, 400), pygame.SRCALPHA)
        text_bg.fill((20, 20, 40, 180))
        screen.blit(text_bg, (WIDTH // 2 - 300, 120))

        for i, line in enumerate(self.instructions):
            text = font.render(line, True, WHITE)
            screen.blit(text, (WIDTH // 2 - text.get_width() // 2, 150 + i * 30))

        back = font.render("按ESC键返回菜单", True, WHITE)
        screen.blit(back, (WIDTH // 2 - back.get_width() // 2, HEIGHT - 50))

    def draw_pause(self, screen):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        screen.blit(overlay, (0, 0))

        title = big_font.render("游戏暂停", True, YELLOW)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, HEIGHT // 2 - 50))

        resume = font.render("按ESC键继续游戏", True, WHITE)
        screen.blit(resume, (WIDTH // 2 - resume.get_width() // 2, HEIGHT // 2 + 50))

        menu = font.render("按M键返回主菜单", True, WHITE)
        screen.blit(menu, (WIDTH // 2 - menu.get_width() // 2, HEIGHT // 2 + 100))
    def run(self):
        running = True
        while running:
            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    # 通用按键
                    if event.key == pygame.K_ESCAPE:
                        if self.game_state == "playing":
                            if self.upgrade_menu:
                                self.upgrade_menu = False
                            else:
                                self.game_state = "paused"
                        elif self.game_state == "paused":
                            self.game_state = "playing"
                        elif self.game_state == "instructions":
                            self.game_state = "menu"

                    # 菜单导航
                    if self.game_state == "menu":
                        if event.key == pygame.K_DOWN:
                            self.menu_selection = (self.menu_selection + 1) % len(self.menu_options)
                        elif event.key == pygame.K_UP:
                            self.menu_selection = (self.menu_selection - 1) % len(self.menu_options)
                        elif event.key == pygame.K_RETURN:
                            if self.menu_selection == 0:  # 开始游戏
                                self.game_state = "playing"
                                self.reset_game()
                            elif self.menu_selection == 1:  # 游戏说明
                                self.game_state = "instructions"
                            elif self.menu_selection == 2:  # 退出游戏
                                running = False

                    # 游戏中的按键
                    if self.game_state == "playing" and not self.upgrade_menu:
                        if event.key == pygame.K_SPACE and not self.game_over:
                            self.player.shoot()
                        elif event.key == pygame.K_u:
                            self.upgrade_menu = not self.upgrade_menu

                    # 暂停菜单中的按键
                    elif self.game_state == "paused":
                        if event.key == pygame.K_m:
                            self.game_state = "menu"

                    # 游戏结束时的按键
                    elif self.game_state == "game_over":
                        if event.key == pygame.K_r:
                            self.game_state = "playing"
                            self.reset_game()
                        elif event.key == pygame.K_m:
                            self.game_state = "menu"

                    # 升级菜单中的按键
                    if self.upgrade_menu:
                        if event.key == pygame.K_1 and self.player.minerals["iron"] >= 50:
                            self.player.upgrades["speed"] += 1
                            self.player.minerals["iron"] -= 50
                        elif event.key == pygame.K_2 and self.player.minerals["iron"] >= 80:
                            self.player.upgrades["laser"] += 1
                            self.player.minerals["iron"] -= 80
                        elif event.key == pygame.K_3 and self.player.minerals["iron"] >= 30:
                            self.player.upgrades["fuel"] += 1
                            self.player.max_fuel += 20
                            self.player.minerals["iron"] -= 30

            # 游戏逻辑更新
            if self.game_state == "playing" and not self.upgrade_menu:
                keys = pygame.key.get_pressed()
                self.player.move(keys)
                self.player.update_lasers()
                self.spawn_asteroids()

                # 更新小行星位置并移除超出边界的
                for asteroid in self.asteroids[:]:
                    if asteroid.move():
                        self.asteroids.remove(asteroid)

                # 检测所有碰撞（包含玩家碰撞检测）
                self.check_collisions()

                # 更新波次和关卡
                self.update_wave()
                self.check_level_up()

                # 燃料消耗和游戏结束检查
                self.player.fuel -= 0.02
                if self.player.fuel <= 0 or self.player.health <= 0:
                    self.game_over = True
                    self.game_state = "game_over"

            # 绘制
            screen.fill(BLACK)

            # 绘制星空背景
            for sx, sy, size in stars:
                pygame.draw.circle(screen, WHITE, (sx, sy), size)

            # 根据游戏状态绘制不同界面
            if self.game_state == "playing":
                # 绘制小行星
                for asteroid in self.asteroids:
                    asteroid.draw(screen)

                # 绘制玩家和UI
                self.player.draw(screen)
                self.draw_ui(screen)

            elif self.game_state == "menu":
                self.draw_menu(screen)

            elif self.game_state == "instructions":
                self.draw_instructions(screen)

            elif self.game_state == "paused":
                for asteroid in self.asteroids:
                    asteroid.draw(screen)
                self.player.draw(screen)
                self.draw_pause(screen)

            elif self.game_state == "game_over":
                for asteroid in self.asteroids:
                    asteroid.draw(screen)
                self.player.draw(screen)
                self.draw_game_over(screen)

            pygame.display.flip()
            clock.tick(60)

if __name__ == "__main__":
    # 确保窗口居中显示
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    Game().run()
    pygame.quit()
    sys.exit()