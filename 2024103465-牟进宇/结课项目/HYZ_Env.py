
import logging
import random
import json
import math
import pandas as pd
import numpy as np

from tqdm import tqdm
from KnowledgeTracing import valid as DKTModel
from mxnet import gpu,cpu

from PromptList.llm_prompt import generate_llm_prompt
from PromptList.llm_prompt import graph_candidateget_prompt_required_information

def load_diff(dataset_filename):
    """
    从指定的 JSON 文件中加载难度数据。

    :param dataset_filename: 包含难度数据的 JSON 文件的路径。
    :return: 包含从 JSON 文件中提取的难度值的列表。
    """
    # 初始化一个空列表，用于存储从文件中读取的难度值
    diff_arr = []
    # 以只读模式打开指定的 JSON 文件
    with open(dataset_filename, 'r') as f:
        # 从文件中加载 JSON 数据，存储为字典
        diff_line = json.load(f)
        # 遍历字典中的每个键值对
        for key, value in diff_line.items():
            # 将字典中的值添加到 diff_arr 列表中
            diff_arr.append(value)
    # 返回包含所有难度值的列表
    return diff_arr

def load_data(dataset_filename):
    """
    从指定的 JSON 格式文件中加载数据。

    :param dataset_filename: 包含数据的 JSON 文件的路径。
    :return: 包含练习题记录、目标知识点集合和学生风格的列表。
    """
    # 将传入的文件名赋值给 true_file，方便后续使用
    true_file = dataset_filename
    # 初始化一个空列表，用于存储从文件中读取的数据
    datas = []
    # 以默认只读模式打开指定文件
    with open(true_file) as f:
        # 使用 tqdm 显示加载数据的进度条
        for line in tqdm(f, "loading data"):
            # 将每行 JSON 字符串解析为 Python 对象
            data = json.loads((line))
            # 检查解析后的数据长度是否为 3
            if len(data) == 3:
                # 若长度为 3，按顺序解包为练习题记录、目标知识点和学生风格
                exercises_record, target, style = json.loads(line)
            else:
                # 若长度不为 3，按特定顺序解包，忽略中间一个元素
                exercises_record, _, target, style = json.loads(line)  # 我们的找起点做法
            # 将练习题记录、目标知识点集合（转换为 set 类型）和学生风格添加到 datas 列表中
            datas.append([exercises_record, set(target), style])
    # 打印加载数据的文件路径
    print(true_file)
    # 返回存储所有数据的列表
    return datas

def load_simdkt(dataset, ctx):
    """
    加载 SimDKT 知识追踪模型。

    :param dataset: 数据集名称，用于确定模型参数。
    :param ctx: 计算设备上下文，如 GPU 或 CPU。
    :return: 初始化好的 DKT 模型实例。
    """
    # 定义 SimDKT 模型的参数关键字参数
    simdkt_params_kwargs = {
        'model_name': 'EmbedDKT',  # 模型名称设置为 EmbedDKT
        'dataset': dataset,  # 指定使用的数据集
        'ctx': ctx,  # 指定计算设备上下文
    }
    # 如果数据集名称属于特定集合，则更新参数中的训练结束轮数
    if dataset in {"junyi_long", "junyi_large", "junyi_session", "junyi_50"}:
        simdkt_params_kwargs.update(
            {'end_epoch': 4}  # 将训练结束轮数设置为 4
        )

    # 初始化 DKT 模型，传入包含参数的 Parameters 实例
    simdkt = DKTModel.DKT(
        params=DKTModel.Parameters(**simdkt_params_kwargs)
    )
    return simdkt

"""
调用包方法：from HYZ_Env import Env


初始化方法：Env = Env(kt,Task2_Test,dif_file,"New",dataset_kc_num,type="test")
其中，kt是知识追踪模型，通过load_simdkt()函数初始化，dataset_filename是数据集路径，dif_file是难度文件路径，"New"是奖励类型，取值可以为Ep、Ep/t和New（new是新任务，具体情况请咨询罗师兄）
dataset_kc_num是知识点个数，type是区别测试还是训练（测试"test"是抽取指定的学生，通过索引实现；训练模式"train"是训练模式，抽取随机学生）

begin是初始化学生，包括重新抽取学生数据，清空推荐路径和奖励。需要在每个episode的开始被调用，调用后会返回学习者初始知识水平

get_target_input()函数会返回当前学习者学习目标

 next_state, reward = env.state_transform(action)，状态转移函数。根据当前动作，进行状态转移，返回奖励和下一时刻状态
 
 get_result()，返回从初始时刻到当前时刻的奖励。上面那个返回的是当前步的奖励变化，这个是返回初始到当前的奖励，即这个是上面的一个汇总。该函数除了奖励，还会返回推荐路径。

"""

class Env(object):
    def __init__(self,kt,dataset_filename,dif_file,reward_type,kc_num,type="train"):
        #存储所有数据
        self.all_data = load_data(dataset_filename)
        self.difficult = load_diff(dif_file)
        # self.difficult = load_diff("data/assistments12/data/diff.json")
        #控制读取数据方式（训练集随机抽取100个学生，测试集加载指定个学生）
        self.type = type
        #明确任务对应的奖励类型
        self.reward_type = reward_type
        #知识追踪模型
        self.kt = kt
        #数据集知识点个数
        self.kc_num = kc_num

        #当前学生数据原始记录
        self.current_student_record = None
        #当前学生目标
        self.current_student_target = None
        #当前学生风格
        self.current_student_style = None
        #当前学生初始知识水平
        self.initial_mastery = None
        #当前学生最大奖励
        self.current_student_max_len = None
        #用于保存kt模型返回的熟练度和隐藏层
        self.current_mastery = None
        self.current_hidden = None
        #用于保存生成的路径
        self.current_path = []
        self.time_list = []
        self.unknown_list = []
        self.difficult_list = []


        self.current_test_stu_id = 0


    def state_transform(self,action):
        """
        根据当前动作进行状态转移，更新学生的知识掌握度并计算奖励。

        Args:
            action (int): 当前选择的题目（对应知识点索引）

        Returns:
            tuple: 下一时刻的知识掌握状态和当前步骤的奖励值
        """
        # 将���前题目的难度添加到难度列表中
        self.difficult_list.append(self.difficult[action])
        # 记录上一时刻的知识掌握度
        old_mastery = self.current_mastery
        # 计算并记录解答当前题目所需的时间，调用 IRT 模型计算
        self.time_list.append(self.irt_time(old_mastery[action],self.difficult[action]))
        # 判断学生解答当前题目是否正确，若对应知识点掌握度大于随机数则认为正确
        is_correct = 1 if self.current_mastery[action]> random.random() else 0
        # 使用知识追踪模型更新学生的知识掌握度和隐藏状态
        temp_current_mastery, self.current_hidden = self.kt([(action, is_correct)], self.current_hidden)
        # 更新当前时刻的知识掌握度为知识追踪模型输出的最后一个结果
        self.current_mastery = temp_current_mastery[-1]
        # 下一时刻的状态即为当前时刻更新后的知识掌握度
        next_state = self.current_mastery
        # 调用 get_reward 方法计算当前步骤的奖励
        reward = self.get_reward(old_mastery,next_state,action)
        # 将当前选择的题目和解答结果添加到推荐路径中
        self.current_path.append([action,is_correct])
        # 若奖励类型为 'unknow'，则将当前步骤的奖励添加到未知奖励列表中
        if self.reward_type == 'unknow':
            self.unknown_list.append(reward)
        # 返回下一时刻的状态和当前步骤的奖励
        return next_state,reward

    def get_reward(self, old_mastery, next_state, action):
        """
        计算当前步骤的奖励值（根据不同的奖励类型）
        
        Args:
            old_mastery: 上一步的知识掌握度向量（各知识点的熟练度）
            next_state: 当前步骤后的知识掌握度向量（各知识点的熟练度）
            action: 当前选择的题目（对应知识点索引）
        
        Returns:
            float: 当前步骤的奖励值
        """
        # 奖励类型：基于目标知识点掌握变化的归一化奖励
        if self.reward_type == 'ep':
            # 将新旧掌握度转换为0/1二值数组（熟练度>0.5视为已掌握）
            old_mastery = np.array(old_mastery)
            next_state = np.array(next_state)
            new_mastery = np.where(next_state > 0.5, 1, 0)  # 新掌握状态（0/1）
            old_mastery = np.where(old_mastery > 0.5, 1, 0)  # 旧掌握状态（0/1）
            mastery_change = new_mastery - old_mastery       # 各知识点掌握变化量（+1/0/-1）
            # 仅关注目标知识点的掌握变化（筛选目标知识点的索引）
            target_mastery_change = [mastery_change[i] for i in self.current_student_target]
            sum_target_mastery_change = sum(target_mastery_change)  # 目标知识点总掌握变化量
            reward = sum_target_mastery_change / self.current_student_max_len  # 归一化（除以最大可能变化量）

        # 奖励类型：目标知识点掌握变化的时间归一化奖励（总变化量/当前步骤时间）
        if self.reward_type == 'ep/t':
            old_mastery = np.array(old_mastery)
            next_state = np.array(next_state)
            new_mastery = np.where(next_state > 0.5, 1, 0)
            old_mastery = np.where(old_mastery > 0.5, 1, 0)
            mastery_change = new_mastery - old_mastery
            target_mastery_change = [mastery_change[i] for i in self.current_student_target]
            sum_target_mastery_change = sum(target_mastery_change)  # 目标知识点总掌握变化量
            reward_ep = sum_target_mastery_change 
            reward = reward_ep / self.time_list[-1]  # 除以当前步骤的时间（IRT模型计算的时间）

        # 奖励类型：结合目标掌握变化与难度平滑度的奖励
        if self.reward_type == 'unknow':
            old_mastery = np.array(old_mastery)
            next_state = np.array(next_state)
            new_mastery = np.where(next_state > 0.5, 1, 0)
            old_mastery = np.where(old_mastery > 0.5, 1, 0)
            mastery_change = new_mastery - old_mastery
            target_mastery_change = [mastery_change[i] for i in self.current_student_target]
            # 目标知识点掌握变化的归一化值 - （当前难度与上一步难度差的平方）
            reward = sum(target_mastery_change) / self.current_student_max_len - (self.difficult_list[-1] - self.difficult_list[-2]) ** 2

        # 奖励类型：整体知识点掌握变化的时间归一化奖励（总变化量/累计时间）
        if self.reward_type == 'New':
            old_mastery = np.array(old_mastery)
            next_state = np.array(next_state)
            new_mastery = np.where(next_state > 0.5, 1, 0)  # 新掌握的知识点（0/1）
            old_mastery = np.where(old_mastery > 0.5, 1, 0)  # 旧掌握的知识点（0/1）
            New_Learned = np.sum(new_mastery == 1)           # 新掌握的总知识点数
            Old_Learned = np.sum(old_mastery == 1)           # 旧掌握的总知识点数
            mastery_change = New_Learned - Old_Learned       # 整体掌握变化量（新增知识点数）
            reward = mastery_change / self.time_list[-1]    # 除以当前步骤的时间

        return reward

    def sample(self):
        index = random.randint(0, len(self.all_data) - 1)
        return index

    def begin(self,current_stu_id = 100):
        #基于当前学生的需要全部初始化
        #当前学生数据原始记录
        self.current_student_record = None
        #当前学生目标
        self.current_student_target = None
        self.current_student_style = None
        #当前学生初始知识水平
        self.initial_mastery = None
        #当前学生最大奖励
        self.current_student_max_len = None
        #用于保存kt模型返回的熟练度和隐藏层
        self.current_mastery = None
        self.current_hidden = None
        #用于保存生成的路径
        self.current_path = []
        self.time_list = []
        self.unknown_list = []
        self.difficult_list = []


        if self.type == "train":
            while True:
                student_id = self.sample()
                exercises_record, target, style = self.all_data[student_id]
                temp_mastery, temp_state = self.kt(exercises_record)
                temp_mastery = temp_mastery[-1]
                np_temp_mastery = np.array(temp_mastery)
                mastery_for_target = np.array([np_temp_mastery[i] for i in target])
                mastery_for_OneHot_target = np.where(mastery_for_target > 0.5, 1, 0)
                if sum(mastery_for_OneHot_target) == len(target):
                    del self.all_data[student_id]
                    continue
                break
        elif self.type == "test":
            while True:
                exercises_record, target, style = self.all_data[current_stu_id]
                temp_mastery, temp_state = self.kt(exercises_record)
                temp_mastery = temp_mastery[-1]
                np_temp_mastery = np.array(temp_mastery)
                mastery_for_target = np.array([np_temp_mastery[i] for i in target])
                mastery_for_OneHot_target = np.where(mastery_for_target > 0.5, 1, 0)
                if sum(mastery_for_OneHot_target) == len(target):
                    del self.all_data[current_stu_id]
                    continue
                break


        self.current_student_record = exercises_record
        self.current_student_target = target
        self.current_student_style = style
        self.current_student_max_len = len(self.current_student_target) - sum(mastery_for_OneHot_target)
        self.current_mastery = temp_mastery
        self.current_hidden = temp_state
        self.initial_mastery = temp_mastery
        self.difficult_list.append(self.difficult[exercises_record[-1][0]])
        return self.current_mastery

    def get_kc_num(self):
        return self.kc_num

    def irt_time(self,ability,difficult):
        ability_diff = ability - difficult
        return 1/(1+math.exp(1.7*ability_diff))

    def get_result(self):
        """
        计算从初始时刻到当前时刻的累计奖励，并返回推荐路径
        
        Returns:
            tuple(float, list): 累计奖励值, 推荐路径（格式为[[题目索引, 是否正确], ...]）
        """
        # 奖励类型：基于目标知识点掌握变化的累计归一化奖励
        if self.reward_type == 'ep':
            # 将初始和当前掌握度转换为0/1二值数组（熟练度>0.5视为已掌握）
            old_mastery = np.array(self.initial_mastery)  # 初始掌握状态
            next_state = np.array(self.current_mastery)   # 当前掌握状态
            new_mastery = np.where(next_state > 0.5, 1, 0)  # 当前掌握状态（0/1）
            old_mastery = np.where(old_mastery > 0.5, 1, 0)  # 初始掌握状态（0/1）
            mastery_change = new_mastery - old_mastery       # 各知识点掌握变化量（+1/0/-1）
            # 仅关注目标知识点的累计变化（筛选目标知识点的索引）
            target_mastery_change = [mastery_change[i] for i in self.current_student_target]
            sum_target_mastery_change = sum(target_mastery_change)  # 目标知识点总变化量
            reward = sum_target_mastery_change / self.current_student_max_len  # 归一化（除以最大可能变化量）

        # 奖励类型：目标知识点掌握变化的累计时间归一化奖励（总变化量/累计时间）
        if self.reward_type == 'ep/t':
            old_mastery = np.array(self.initial_mastery)
            next_state = np.array(self.current_mastery)
            new_mastery = np.where(next_state > 0.5, 1, 0)
            old_mastery = np.where(old_mastery > 0.5, 1, 0)
            mastery_change = new_mastery - old_mastery
            target_mastery_change = [mastery_change[i] for i in self.current_student_target]
            sum_target_mastery_change = sum(target_mastery_change)  # 目标知识点总变化量
            reward_ep = sum_target_mastery_change 
            reward = reward_ep / sum(self.time_list)  # 除以所有步骤的累计时间（IRT模型计算的时间）

        # 奖励类型：未知类型的累计奖励（直接累加每步的未知奖励值）
        if self.reward_type == 'unknow':
            reward = sum(self.unknown_list)  # 累加每一步记录的未知奖励值

        # 奖励类型：整体知识点掌握变化的累计时间归一化奖励（总变化量/累计时间）
        if self.reward_type == 'New':
            old_mastery = np.array(self.initial_mastery)
            next_state = np.array(self.current_mastery)
            new_mastery = np.where(next_state > 0.5, 1, 0)  # 当前掌握的知识点（0/1）
            old_mastery = np.where(old_mastery > 0.5, 1, 0)  # 初始掌握的知识点（0/1）
            New_Learned = np.sum(new_mastery == 1)           # 当前掌握的总知识点数
            Old_Learned = np.sum(old_mastery == 1)           # 初始掌握的总知识点数
            mastery_change = New_Learned - Old_Learned       # 整体掌握变化量（新增知识点数）
            reward = mastery_change / sum(self.time_list)    # 除以所有步骤的累计时间

        return reward, self.current_path

    def get_pnode(self):
        temp = self.current_student_record + self.current_path
        return temp[-1][0]


    def get_target(self):
        return self.current_student_target

    def get_style(self):
        return self.current_student_style[0]

    def get_target_input(self):
        input_list = [0.0]*self.kc_num

        for i in self.current_student_target:
            input_list[i] = 1

        return input_list


    def begin_test(self):
        #基于当前学生的需要全部初始化
        #当前学生数据原始记录
        self.current_student_record = None
        #当前学生目标
        self.current_student_target = None
        self.current_student_style = None
        #当前学生初始知识水平
        self.initial_mastery = None
        #当前学生最大奖励
        self.current_student_max_len = None
        #用于保存kt模型返回的熟练度和隐藏层
        self.current_mastery = None
        self.current_hidden = None
        #用于保存生成的路径
        self.current_path = []
        self.time_list = []
        self.unknown_list = []
        self.difficult_list = []


        while True:
            if self.current_test_stu_id == len(self.all_data)-1:
                return None

            exercises_record, target, style = self.all_data[self.current_test_stu_id]
            temp_mastery, temp_state = self.kt(exercises_record)
            temp_mastery = temp_mastery[-1]
            np_temp_mastery = np.array(temp_mastery)
            mastery_for_target = np.array([np_temp_mastery[i] for i in target])
            mastery_for_OneHot_target = np.where(mastery_for_target > 0.5, 1, 0)
            if sum(mastery_for_OneHot_target) == len(target):
                del self.all_data[self.current_test_stu_id]
                continue
            break

        self.current_test_stu_id = self.current_test_stu_id + 1
        self.current_student_record = exercises_record
        self.current_student_target = target
        self.current_student_style = style
        self.current_student_max_len = len(self.current_student_target) - sum(mastery_for_OneHot_target)
        self.current_mastery = temp_mastery
        self.current_hidden = temp_state
        self.initial_mastery = temp_mastery
        self.difficult_list.append(self.difficult[exercises_record[-1][0]])
        return self.current_mastery

    def resert_student_id(self):
        self.current_test_stu_id = 0
    


if __name__ == '__main__':
    ###生成prompt###
    print("生成prompt")