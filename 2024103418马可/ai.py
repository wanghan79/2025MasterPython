from collections import deque
from config import GRID_WIDTH, GRID_HEIGHT, UP, DOWN, LEFT, RIGHT

def find_path_bfs(start_pos, food_pos, snake_body, obstacles):
    """
    使用广度优先搜索 (BFS) 寻找从蛇头到食物的最短路径。
    :param start_pos: 蛇头坐标 (x, y)
    :param food_pos: 食物坐标 (x, y)
    :param snake_body: 蛇身体坐标列表，作为障碍物
    :return: 到达食物的移动方向列表，如 [RIGHT, RIGHT, UP]。如果找不到路径则返回 None。
    """
    all_obstacles = set(snake_body) | set(obstacles)


    # 初始化队列，每个元素是 (当前位置, 到达此位置的路径)
    queue = deque([(start_pos, [])])


    # 访问过的节点集合，防止重复搜索
    visited = {start_pos}

    while queue:
        current_pos, path = queue.popleft()

        # 如果找到食物，返回路径
        if current_pos == food_pos:
            return path

        # 探索四个方向的邻居
        for move_dir in [UP, DOWN, LEFT, RIGHT]:
            next_x = current_pos[0] + move_dir[0]
            next_y = current_pos[1] + move_dir[1]
            next_pos = (next_x, next_y)

            # 检查邻居是否合法
            if (0 <= next_x < GRID_WIDTH and
                0 <= next_y < GRID_HEIGHT and
                next_pos not in all_obstacles and
                next_pos not in visited):
                
                visited.add(next_pos)
                new_path = path + [move_dir]
                queue.append((next_pos, new_path))

    # 如果队列为空仍未找到食物，说明没有路径   
    return None