import random
from typing import Dict, Tuple, Set, List
import networkx as nx
from tqdm import tqdm
import heapq


def generate_snapshots(
    G_gamma: nx.DiGraph,
    R: int,
    progress_callback=None  # 添加可选的回调函数参数
) -> List[nx.DiGraph]:
    """
    生成 R 个快照。

    Parameters:
    - G_gamma (nx.DiGraph): 带有激活概率的社交图。
    - R (int): 快照数量。
    - progress_callback (callable, optional): 进度回调函数。

    Returns:
    - snapshots (List[nx.DiGraph]): 生成的快照列表。
    """
    snapshots = []
    for snapshot_idx in range(R):
        # Step 4: Generate g'i
        g_prime = G_gamma.copy()
        edges_to_remove = []
        for u, v, data in g_prime.edges(data=True):
            ap = data.get('ap', 1.0)
            if random.random() >= ap:  # 注意这里是>=，表示以1-ap的概率删除
                edges_to_remove.append((u, v))
        g_prime.remove_edges_from(edges_to_remove)
        
        # Step 5: Generate gi
        gi = g_prime.copy()
        active_nodes = set()
        for u, v in g_prime.edges():
            active_nodes.add(u)
            active_nodes.add(v)

        for node in active_nodes:
            for neighbor in G_gamma.successors(node):
                if neighbor not in active_nodes:
                    gi.add_edge(node, neighbor, ap=G_gamma[node][neighbor]['ap'])
        
        snapshots.append(gi)
        
        # 如果提供了回调函数，则调用它
        if progress_callback:
            progress_callback()
            
    return snapshots

def find_seeds_in_candidates(
    G: nx.DiGraph,
    topic_distribution: List[float],
    k: int,
    R: int,
    Sc: Set
) -> Set:
    """
    Algorithm 3: FindSeedsInCandidates
    Finds a size-k seed set S from candidate seeds Sc based on the given social graph G,
    topic distribution gamma, number of snapshots R, and number of seeds k.

    Parameters:
    - G (nx.DiGraph): The social graph where nodes represent individuals and edges represent connections.
    - topic_distribution (List[float]): The topic distribution of the query.
    - k (int): Number of seeds to find.
    - R (int): Number of snapshots to generate.
    - Sc (Set): Set of candidate seed nodes.

    Returns:
    - S (Set): A size-k seed set for the given topic.
    """
    
    # Initialize the seed set S
    S: Set = set()
    
    # Step 2: Compute activation probabilities ap(u, v | gamma) for each edge in G
    G_gamma = G.copy()
    for u, v in G_gamma.edges():
        G_gamma[u][v]['ap'] = sum([topic_distribution[i] * G.nodes[v]['topics'][i] for i in range(len(topic_distribution))])
    
    # 使用嵌套的tqdm来显示总体进度
    pbar = tqdm(total=R * 3 + k, desc="查找种子节点-SG", leave=False)
    
    # Step 3: Generate R snapshots based on the activation probabilities
    snapshots = generate_snapshots(G_gamma, R, lambda: pbar.update(1))
    
    # Precompute reachability for all nodes in all snapshots to speed up
    # This can be memory intensive for large graphs and R
    # Consider optimizing or computing on-the-fly for large instances
    reachability_snapshots: List[Dict] = []
    for gi in snapshots:
        reachability = {}
        for node in gi.nodes():
            reachable = nx.descendants(gi, node)
            reachable.add(node)  # Include the node itself
            reachability[node] = reachable
        reachability_snapshots.append(reachability)
        pbar.update(1)
    
    # Step 6: Iteratively select k seeds
    for seed_idx in range(k):
        Mg = {v: 0 for v in Sc if v not in S}
        
        # Step 7-11: Compute the marginal gain for each candidate seed
        for j in range(R):
            gi_reach = reachability_snapshots[j]
            if S:
                # Union of reachability from all seeds in S
                reachable_S = set()
                for s in S:
                    reachable_S.update(gi_reach.get(s, set()))
            else:
                reachable_S = set()
            
            for v in Mg.keys():
                # Reachable from S ∪ {v}
                if v in gi_reach:
                    reachable_Sv = reachable_S.union(gi_reach[v])
                else:
                    reachable_Sv = reachable_S.copy()
                
                # The marginal gain is the number of newly reachable nodes by adding v
                Mg[v] += len(reachable_Sv) - len(reachable_S)
        
        if not Mg:
            # No more candidates to select
            break
        
        # Step 12: Select the candidate with the highest marginal gain
        best_v = max(Mg, key=Mg.get)
        S.add(best_v)
        # print(f"Selected seed {seed_idx + 1}: {best_v} with marginal gain {Mg[best_v]}")
        pbar.update(1)
    
    # Step 13: Return the final seed set
    pbar.close()
    return S

def find_seeds_v1(
    G: nx.DiGraph,
    topic_distribution: List[float],
    k: int,
    R: int,
    Sc: Set[str]
) -> Set[str]:
    """
    RSS-v1: Uses the v1 optimization for seed selection.

    Parameters:
    - G (nx.DiGraph): 社交图。
    - topic_distribution (List[float]): 查询的主题分布。
    - k (int): 需要选择的种子数量。
    - R (int): 快照数量。
    - Sc (Set[str]): 候选种子节点集。

    Returns:
    - S (Set[str]): 选择的种子节点集。
    """
    # 初始化种子集 S 和优先队列（最大堆）
    S: Set[str] = set()
    heap = []
    last_mg: Dict[str, int] = {}

    # Step 2: 计算激活概率
    G_gamma = G.copy()
    for u, v in G_gamma.edges():
        G_gamma[u][v]['ap'] = sum([topic_distribution[i] * G.nodes[v]['topics'][i] for i in range(len(topic_distribution))])

    # 添加总体进度条
    pbar = tqdm(total=R * 3 + k, desc="查找种子节点-v1", leave=False)
    
    # Step 3: 生成 R 个快照
    snapshots = generate_snapshots(G_gamma, R, lambda: pbar.update(1))

    # 预计算每个快照中每个节点的可达节点集
    reachability_snapshots: List[Dict[str, Set[str]]] = []
    for gi in snapshots:
        reachability = {}
        for node in gi.nodes():
            reachable = nx.descendants(gi, node)
            reachable.add(node)
            reachability[node] = reachable
        reachability_snapshots.append(reachability)
        pbar.update(1)

    # 初始阶段：计算每个候选节点的初始边际增益
    for v in Sc:
        mg = 0
        for j in range(R):
            gi_reach = reachability_snapshots[j]
            mg += len(gi_reach.get(v, set()))
        last_mg[v] = mg
        heapq.heappush(heap, (-mg, v))  # 使用负值构建最大堆

    # 迭代选择 k 个种子
    for seed_idx in range(k):
        if not heap:
            break

        # 弹出堆顶元素
        current_mg, v = heapq.heappop(heap)
        current_mg = -current_mg

        # 计算 S 的可达节点集
        reachable_S = set()
        if S:
            for s in S:
                for j in range(R):
                    reachable_S.update(reachability_snapshots[j].get(s, set()))
        
        # 重新计算 v 的边际增益
        new_mg = 0
        for j in range(R):
            gi_reach = reachability_snapshots[j]
            reachable_Sv = reachable_S.union(gi_reach.get(v, set()))
            new_mg += len(reachable_Sv)

        # 计算真实的边际增益
        if S:
            previous_mg = sum(len(reachability_snapshots[j].get(s, set())) for s in S for j in range(R))
            actual_mg = new_mg - previous_mg
        else:
            actual_mg = new_mg

        # 如果边际增益与缓存一致，则选择它
        if actual_mg == last_mg[v]:
            S.add(v)
            # print(f"Selected seed {seed_idx + 1}: {v} with marginal gain {actual_mg}")
            pbar.update(1)
        else:
            # 更新边际增益并重新插入堆
            last_mg[v] = actual_mg
            heapq.heappush(heap, (-actual_mg, v))

    pbar.close()
    return S

def find_seeds_degree(
    G: nx.DiGraph,
    topic_distribution: List[float],
    k: int,
    R: int,
    Sc: Set[str]
) -> Set[str]:
    """
    RSS-Degree: Selects seeds based on the out-degree in the original graph.

    Parameters:
    - G (nx.DiGraph): 社交图。
    - topic_distribution (List[float]): 查询的主题分布。
    - k (int): 需要选择的种子数量。
    - R (int): 快照数量。
    - Sc (Set[str]): 候选种子节点集。

    Returns:
    - S (Set[str]): 选择的种子节点集。
    """
    # 初始化种子集
    S: Set[str] = set()

    # Step 2: 计算激活概率
    G_gamma = G.copy()
    for u, v in G_gamma.edges():
        G_gamma[u][v]['ap'] = sum([topic_distribution[i] * G.nodes[v]['topics'][i] for i in range(len(topic_distribution))])

    # 添加���体进度条
    pbar = tqdm(total=R * 2 + k, desc="查找种子节点-Degree", leave=False)
    
    # Step 3: 生成 R 个快照
    snapshots = generate_snapshots(G_gamma, R, lambda: pbar.update(1))
    pbar.update(R)  # 由于Degree策略不需要预计算可达性，直接更新R次

    # 选择候选节点中度数最高的 k 个节点
    degree_dict = {v: G.out_degree(v) for v in Sc}
    # 按照度数降序排序
    sorted_candidates = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)
    # 选择前 k 个节点
    for i in range(min(k, len(sorted_candidates))):
        v, deg = sorted_candidates[i]
        S.add(v)
        # print(f"Selected seed {i + 1}: {v} with degree {deg}")
        pbar.update(1)

    pbar.close()
    return S


def preprocess(G, topics, K, R):
    """
    预处理阶段：为每个话题生成种子集合
    :param G: 社交图
    :param topics: 话题列表
    :param K: 每个话题的种子节点数
    :param R: 生成快照的数量
    :return: 每个话题对应的种子节点集合
    """
    seeds_per_topic = {}
    # 添加进度条
    for topic in tqdm(topics, desc="预处理话题"):
        topic_distribution = [1 if i == topic else 0 for i in range(len(topics))]
        seeds_per_topic[topic] = find_seeds_in_candidates(G, topic_distribution, K, R, G.nodes())
    return seeds_per_topic


def monte_carlo_snapshot(G, topic_distribution):
    """
    生成 Monte Carlo 快照，同时标记活跃节点和被告知节点
    :param G: 社交图
    :param topic_distribution: 查询的特定话题分布
    :return: 活跃节点集合和被告知节点集合
    """
    active_nodes = set()
    informed_nodes = set()
    for u, v in G.edges():
        # 计算边 (u, v) 的激活概率
        activation_prob = sum([topic_distribution[i] * G.nodes[v]['topics'][i] for i in range(len(topic_distribution))])
        
        # 判断节点状态
        if random.random() < activation_prob:
            active_nodes.add(v)
        elif u in active_nodes:
            informed_nodes.add(v)
            
    return active_nodes, informed_nodes

def rss(G, query, seeds_per_topic, k, R):
    """
    RSS 主函数，根据查询动态合并种子集合并选取最终的种子节点
    :param G: 社交图
    :param query: 查询 (包括话题分布和种子节点数量)
    :param seeds_per_topic: 每个话题的种子集合
    :param k: 种子节点数
    :param R: Monte Carlo 快照数量
    :return: 最终的种子节点集合
    """
    topic_distribution, seed_count = query
    # 初始化候选合
    candidates = set()
    # 合并各个话题的种子节点集合
    for topic, weight in enumerate(topic_distribution):
        if weight > 0:
            candidates.update(seeds_per_topic[topic])

    # 从候选集中选择最终的种子节点
    final_seeds = find_seeds_in_candidates(G, topic_distribution, seed_count, R, candidates)
    final_seeds_v1 = find_seeds_v1(G, topic_distribution, seed_count, R, candidates)
    final_seeds_degree = find_seeds_degree(G, topic_distribution, seed_count, R, candidates)
    return final_seeds, final_seeds_v1, final_seeds_degree    

def im(G, num_topics, topic_distribution, seed_count):
    # 1. 预处理阶段：生成每个话题的初步种子集合
    topics = list(range(num_topics))  # 假设话题数为 num_topics
    K = 50  # 每个话题的种子节点数量
    R = 100  # Monte Carlo 快照数量
    seeds_per_topic = preprocess(G, topics, K, R)

    # 2. 查询阶段：根据特定查询选择最终的种子节点
    query = (topic_distribution, seed_count)

    # 执行 RSS 算法
    final_seeds, final_seeds_v1, final_seeds_degree  = rss(G, query, seeds_per_topic, seed_count, R)
    
    return final_seeds, final_seeds_v1, final_seeds_degree 
    

