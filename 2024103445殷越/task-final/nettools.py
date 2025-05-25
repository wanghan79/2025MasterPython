import random

def monte_carlo_snapshot(G, topic_distribution, seeds):
    """
    基于种子节点生成 Monte Carlo 快照，模拟信息传播过程。
    :param G: 社交图
    :param topic_distribution: 查询的话题分布
    :param seeds: 初始种子节点集合
    :return: 最终的活跃节点集合
    """
    active_nodes = set(seeds)
    new_active = set(seeds)

    # 模拟信息传播
    while new_active:
        next_active = set()
        for u in new_active:
            for v in G.successors(u):  # 只考虑出度邻居
                if v not in active_nodes:
                    # 计算传播激活概率
                    activation_prob = sum([topic_distribution[i] * G.nodes[v]['topics'][i] for i in range(len(topic_distribution))])
                    if random.random() < activation_prob:
                        next_active.add(v)
                        active_nodes.add(v)
        new_active = next_active

    return active_nodes

def calculate_coverage(G, seeds, topic_distribution, R):
    """
    计算给定种子节点集合的平均覆盖率
    :param G: 社交图
    :param seeds: 种子节点集合
    :param topic_distribution: 查询的话题分布
    :param R: Monte Carlo 模拟次数
    :return: 平均覆盖率
    """
    total_active_nodes = 0
    for _ in range(R):
        active_nodes = monte_carlo_snapshot(G, topic_distribution, seeds)
        total_active_nodes += len(active_nodes)
    
    # 返回平均覆盖率
    return total_active_nodes / R
