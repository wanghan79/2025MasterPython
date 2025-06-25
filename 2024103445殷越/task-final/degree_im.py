import networkx as nx

def degree(G, query, k):
    """
    选择与查询话题相关的出度最高的节点。
    :param G: 社交图
    :param query: 查询，包含话题分布向量
    :param k: 所需种子节点数
    :return: 出度最高的种子节点集合
    """
    topic_distribution = query[0]  # 查询的多话题分布向量
    node_scores = {}

    # 计算每个节点的与查询话题相关的出度加权值
    for node in G.nodes():
        weighted_out_degree = 0
        for neighbor in G.successors(node):  # 只计算出度邻居
            # 计算邻居节点与查询话题的相关度
            topic_relevance = sum([topic_distribution[i] * G.nodes[neighbor]['topics'][i] for i in range(len(topic_distribution))])
            weighted_out_degree += topic_relevance
        
        node_scores[node] = weighted_out_degree

    # 按照加权出度排序并选择前 k 个节点作为种子节点
    sorted_nodes = sorted(node_scores, key=node_scores.get, reverse=True)
    return set(sorted_nodes[:k])

if __name__ == "__main__":  
    # 示例使用
    query_topic_distribution = [0.4, 0.3, 0.3]  # 示例查询的话题分布
    k = 10  # 所需种子节点数量
    query = (query_topic_distribution, k)

    # 假设 G 已定义
    #seed_nodes = ta_degree(G, query, k)
    #print("选择的种子节点:", seed_nodes)
