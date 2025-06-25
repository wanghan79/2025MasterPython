import random


class NormalDistribution:
    """生成正态分布随机数"""

    def random(self):
        return random.normalvariate(0, 1)  # 均值0，标准差1


class CustomRandom:
    """自定义随机整数生成器"""

    def random(self, a, b):
        return random.randint(a, b)


class NoRandomMethod:
    """无random方法的自定义类"""

    def __init__(self, value=0):
        self.value = value


class ErrorClass:
    """用于测试异常处理的类"""

    def __init__(self):
        raise ValueError("初始化失败")