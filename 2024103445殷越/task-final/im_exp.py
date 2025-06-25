import numpy as np
import random
import time
from im import im
from degree_im import ta_degree
from nettools import calculate_coverage
import networkx as nx
from lfs_aminer_datasets_loader import load_aminer_dataset
from tqdm import tqdm

def generate_random_query_by_keywords(num_topics, q, k):
    """
    生成一个查询，包括随机选择的关键词集合（即话题组合）和种子节点数量。
    :param num_topics: 总话题数量
    :param q: 查询包含的关键词数量
    :param k: 种子节点数量
    :return: 查询的关键词集合和种子节点数量
    """
    selected_topics = random.sample(range(num_topics), q)
    topic_distribution = np.zeros(num_topics)
    weights = np.random.rand(q)
    weights /= weights.sum()
    for i, topic in enumerate(selected_topics):
        topic_distribution[topic] = weights[i]
    return topic_distribution, k

def experiment_with_im(G, num_topics, R=100):
    """
    执行实验，比较各算法在不同查询条件下的覆盖率和运行时间。
    :param G: 社交图
    :param num_topics: 话题总数量
    :param R: Monte Carlo 模拟次数
    :return: 结果列表，包含覆盖率和运行时间
    """
    results = create_empty_result_dict()
    
    for q in range(1, 6):
        for k in range(10, 51, 10):
            experiment_with_im_once(G, num_topics, R, results, q, k)

    return results

def experiment_with_im_once(G, num_topics, R, results, q, k):
    im_coverages = []
    im_times = []
    ta_degree_coverages = []
    ta_degree_times = []
    im_coverages_v1 = []
    im_times_v1 = []
    im_coverages_degree = []
    im_times_degree = []
            
    # 随机生成 50 个查询
    for _ in tqdm(range(50),desc="生成查询"):
        topic_distribution, seed_count = generate_random_query_by_keywords(num_topics, q, k)
                
        # 调用 im 函数并计算覆盖率和时间
        start_time = time.time()
        final_seeds_im, final_seeds_im_v1, final_seeds_im_degree = im(G, num_topics, topic_distribution, seed_count)
        im_time = time.time() - start_time
        im_coverage = calculate_coverage(G, final_seeds_im, topic_distribution, R)
        im_coverage_v1 = calculate_coverage(G, final_seeds_im_v1, topic_distribution, R)
        im_coverage_degree = calculate_coverage(G, final_seeds_im_degree, topic_distribution, R)
                
        im_coverages.append(im_coverage)
        im_times.append(im_time)
        im_coverages_v1.append(im_coverage_v1)
        im_times_v1.append(im_time)
        im_coverages_degree.append(im_coverage_degree)
        im_times_degree.append(im_time)
                
        # 调用 TA-Degree 基准算法
        start_time = time.time()
        final_seeds_ta_degree = ta_degree(G, (topic_distribution, seed_count), seed_count)
        ta_degree_time = time.time() - start_time
        ta_degree_coverage = calculate_coverage(G, final_seeds_ta_degree, topic_distribution, R)
                
        ta_degree_coverages.append(ta_degree_coverage)
        ta_degree_times.append(ta_degree_time)
            
    # 记录平均覆盖率和时间
    results['IM']['coverage'].append(np.mean(im_coverages))
    results['IM']['time'].append(np.mean(im_times))
    results['IM-v1']['coverage'].append(np.mean(im_coverages_v1))
    results['IM-v1']['time'].append(np.mean(im_times_v1))
    results['IM-Degree']['coverage'].append(np.mean(im_coverages_degree))
    results['IM-Degree']['time'].append(np.mean(im_times_degree))
    results['TA-Degree']['coverage'].append(np.mean(ta_degree_coverages))
    results['TA-Degree']['time'].append(np.mean(ta_degree_times))

def create_empty_result_dict():
    return {
        'IM': {'coverage': [], 'time': []},
        'TA-Degree': {'coverage': [], 'time': []},
        'IM-v1': {'coverage': [], 'time': []},
        'IM-Degree': {'coverage': [], 'time': []}
    }

if __name__ == "__main__":
    # 执行实验并获取结果
    graph,num_topics = load_aminer_dataset('./data/lfs.aminer.cn/graphs_pubs')  # 使用实际的社交图
    #results = experiment_with_im(graph, num_topics)
    
    results = create_empty_result_dict()
    experiment_with_im_once(graph, num_topics, 100, results, 3, 10)
    

    # 打印实验结果
    print("IM 平均覆盖率:", results['IM']['coverage'])
    print("IM 平均运行时间:", results['IM']['time'])
    print("IM-v1 平均覆盖率:", results['IM-v1']['coverage'])
    print("IM-v1 平均运行时间:", results['IM-v1']['time'])
    print("IM-Degree 平均覆盖率:", results['IM-Degree']['coverage'])
    print("IM-Degree 平均运行时间:", results['IM-Degree']['time'])
    print("TA-Degree 平均覆盖率:", results['TA-Degree']['coverage'])
    print("TA-Degree 平均运行时间:", results['TA-Degree']['time'])
