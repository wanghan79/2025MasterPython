import ast  # 用于安全地解析字符串形式的列表
# 生成prompt
def generate_llm_prompt(goals_kcs, all_items, mastery_vector, current_item, candidates):
    """
    生成结构清晰的英文提示词，用于学习路径推荐的 LLM 推理。

    参数：
        goals (list[str]): 学习目标列表，例如 ['A', 'B', 'C']
        all_items (list[str]): 所有学习项，例如 ['A', 'B', 'C', 'D', 'E', 'F']
        mastery_vector (list[float]): 各学习项对应的掌握程度，例如 [0.1, 0.5, 0.2, 0.3, 0.0, 0.0]
        current_item (str): 当前正在学习的项目，例如 'D'
        candidates (list[str]): 候选学习项，例如 ['A', 'B', 'E', 'F']

    返回：
        prompt (str): 英文提示词，适用于大语言模型
    """
    # goals_str = ', '.join(goals)
    # all_items_str = ', '.join(all_items)
    # mastery_str = mastery_vector
    # candidates_str = ', '.join(candidates)

    prompt = (
        f"You are performing a learning path recommendation task. "
        f"Your goal is to recommend learning items to the learner step by step in order to help them maximize "
        f"their mastery of the learning targets: {goals_kcs}, where each mastery level ranges from 0 to 1.\n"
        f"The learner’s current mastery levels over all learning items {all_items} are represented by the vector: "
        f"{mastery_vector}. Each value in this vector indicates the learner’s current level of understanding for the corresponding item.\n"
        f"The learner is currently studying item {current_item}.\n"
        f"You are given the following candidate actions: {candidates}.\n"
        f"Please follow these steps:\n"
        f"1. Briefly analyze the learner’s current mastery levels of the target items {goals_kcs}.\n"
        f"2. Consider the potential benefit of each candidate item in improving the learner’s mastery of the target goals.\n"
        f"3. Identify which candidate item would be most helpful for achieving the learning targets.\n"
        f"4. Output only the selected learning item as your final decision.\n"
        f"Please reason step by step before making your final recommendation.\n"
        f"** Only output the name of the recommended item.Do not output any explanation, reasoning process, or additional text.**"
    )

    return prompt

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
# 读取推荐数据的某一行数据中的第一个元素和第二个元素，即历史答题记录和学习目标
def read_history_records(file_path, line_number):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if line_number < len(lines):
            line_data = ast.literal_eval(lines[line_number])  # 将字符串解析为Python对象
            learning_history_records = line_data[0]
            learning_goals = line_data[1]
            return learning_history_records, learning_goals
        else:
            print("指定的行号超出范围")
# 获取构造prompt所需的相关信息, pnode为当前学习的学习项, line_number为读取的推荐数据集的行号
def get_prompt_required_information(dataset, pnode, line_number):
    """
    获取生成 LLM 提示词所需的相关信息。

    参数:
        dataset (str): 数据集名称，例如 'aicfe'。
        pnode (int): 当前学习的知识点编号。
        line_number (int): 读取的推荐数据集的行号。

    返回:
        tuple: 包含目标知识点名称、所有知识点名称、知识掌握向量、当前知识点名称和候选知识点名称的元组。
    """
    # 应用示例，调用 read_history_records 函数读取指定行的推荐数据
    # 第一个元素为历史学习记录，第二个元素为学习目标
    history_records, goals = read_history_records(f'data/{dataset}_50/data/rec_train', line_number)
    # 打印第 line_number 行推荐数据的历史记录和学习目标
    print(f'第{line_number}行推荐数据的历史记录和学习目标为：', history_records, goals)
    # 定义数据集与知识点映射的字典
    dataset_kcs_map = {
        # 键为数据集名称，值应为对应数据集的知识点列表，目前变量未定义
        "junyi": junyi_kcs,
        "assistments12": assistments12_kcs,
        "aicfe": aicfe_kcs
    }
    # 根据数据集名称从映射字典中获取对应的知识点列表
    kcs_list = dataset_kcs_map.get(dataset)
    # 目标知识点编号对应的知识点名称，通过列表索引获取
    goals_kcs = [kcs_list[i] for i in goals]
    # 所有知识点的名称，直接赋值为获取到的知识点列表
    all_items = kcs_list


    # 加载训练好的 DKT（Deep Knowledge Tracing）模型
    # 传入数据集名称和 CPU 设备信息，目前 load_simdkt 和 cpu 函数未定义
    dkt = load_simdkt(f"{dataset}_50", cpu(0))
    # 调用 DKT 模型，传入历史学习记录，获取知识掌握情况
    mastery, _ = dkt(history_records)
    # 当前知识水平，取最后一个时间步的掌握情况，TODO 表示知识水平是变化的，后续可能需要更新
    mastery_vector = mastery[-1]
    # 当前知识点的名称，通过当前知识点编号从知识点列表中获取
    current_item = kcs_list[pnode]
    # 创建图对象，用于后续候选集的生成，目前 Graph 类未定义
    graph = Graph(dataset=f"{dataset}_50")

    # 调用 graph_candidate 方法，获取候选集
    # 传入图对象、当前知识掌握向量和当前知识点编号，目前 graph_candidate 函数未定义
    candidates, _ = graph_candidate(
        graph=graph,
        mastery=mastery_vector,
        pnode=pnode
    )
    # 候选项知识点编号对应的知识点名称，通过列表索引获取
    candidates_kcs = [kcs_list[i] for i in candidates]
    # 返回生成 LLM 提示词所需的相关信息
    return goals_kcs, all_items, mastery_vector, current_item, candidates_kcs
   #tuple: 包含目标知识点名称、所有知识点名称、知识掌握向量、当前知识点名称和候选知识点名称的元组。

# 获取所需的相关信息后，生成prompt
#prompt = generate_llm_prompt(*get_prompt_required_information('aicfe', 3, 2))
#print(prompt)
if __name__ == '__main__':
    # 示例数据
    goals_kcs = ['A', 'B', 'C']
    all_items = ['A', 'B', 'C', 'D', 'E', 'F']
    mastery_vector = [0.1, 0.5, 0.2, 0.3, 0.0, 0.0]
    current_item = 'D'
    candidates = ['A', 'B', 'E', 'F']

    # 生成提示词
    prompt = generate_llm_prompt(goals_kcs, all_items, mastery_vector, current_item, candidates)
    print(prompt)