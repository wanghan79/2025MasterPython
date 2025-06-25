import networkx as nx
import numpy as np
import os


def read_citation_file(file_path):
    """
    读取单个主题的引用网络文件，解析顶点和边。
    :param file_path: 数据文件路径
    :return: 包含节点和边的图结构 (networkx.DiGraph)
    """
    # 创建图
    G = nx.DiGraph()
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        reading_vertices = False
        reading_edges = False
        
        for line in lines:
            line = line.strip()
            
            # 检查顶点和边部分的开始标志
            if line.startswith('*Vertices'):
                reading_vertices = True
                reading_edges = False
                continue
            elif line.startswith('*Edges'):
                reading_vertices = False
                reading_edges = True
                continue
            
            # 解析顶点信息
            if reading_vertices:
                try:
                    # 使用引号作为分隔符来正确处理带空格的标题
                    if '"' in line:
                        # 分割第一个空格（节点ID）
                        node_id_part, rest = line.split(" ", 1)
                        paper_id = int(node_id_part)
                        
                        # 提取引号中的标题和引用次数
                        title_start = rest.find('"') + 1
                        title_end = rest.rfind('"')
                        title = rest[title_start:title_end]
                        
                        # 获取最后的引用次数
                        num_cited = int(rest[title_end + 1:].strip())
                    else:
                        parts = line.split(" ", 2)
                        paper_id = int(parts[0])
                        title = parts[1]
                        num_cited = int(parts[2]) if len(parts) > 2 else 0
                        
                    # 添加节点到图中
                    G.add_node(paper_id, title=title, num_cited=num_cited)
                except (ValueError, IndexError) as e:
                    print(f"Warning: 跳过格式错误的行: {line}")
                    continue
            
            # 解析边信息
            elif reading_edges:
                # 分离出 source, target 和权重
                parts = line.split()
                source = int(parts[0])
                target = int(parts[1])
                weight = int(parts[2]) if len(parts) > 2 else 1  # 边的权重（默认为1）
                
                # 添加边到图中
                G.add_edge(source, target, weight=weight)
    
    return G

def read_citation_file_raw(file_path):
    """
    读取引用网络数据文件，解析顶点和边信息。
    :param file_path: 数据文件路径
    :return: 包含节点和边的图结构 (networkx.DiGraph)
    """
    # 创建一个有向图
    G = nx.DiGraph()
    
    edges_count = 0
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        reading_vertices = False
        reading_edges = False
        
        for line in lines:
            line = line.strip()
            
            # 读取节点部分
            if line.startswith('*Vertices'):
                reading_vertices = True
                reading_edges = False
                continue
            elif line.startswith('*Edges'):
                reading_vertices = False
                reading_edges = True
                continue
            
            # 解析节点信息
            if reading_vertices and line:
                # 分离出节点的 id、标题、年份、出版地点、作者信息
                parts = line.split("\t")
                node_attrs = {'title': parts[1]}  # title 是必需的
                
                # 可选属性
                attrs_map = {
                    2: ('year', lambda x: int(x)),
                    3: ('venue', str),
                    4: ('authors', lambda x: [a.strip() for a in x.split(",") if a.strip()])
                }
                
                for idx, (key, converter) in attrs_map.items():
                    if len(parts) > idx:
                        try:
                            value = converter(parts[idx])
                            if value:  # 忽略空值
                                node_attrs[key] = value
                        except (ValueError, IndexError):
                            continue
                
                # 添加节点
                G.add_node(int(parts[0]), **node_attrs)
            
            # 解析边信息
            elif reading_edges and line:
                # 分离出 source_id、target_id 和权重
                parts = line.split()
                source_id = int(parts[0])
                target_id = int(parts[1])
                weight = int(parts[2]) if len(parts) > 2 else 1  # 边的权重（默认为1）
                
                if(G.has_edge(source_id, target_id)):
                    G[source_id][target_id]['weight'] += weight
                    edges_count+=1
                else:
                    # 添加边到图中
                    G.add_edge(source_id, target_id, weight=weight)
                    edges_count+=1
                    
    print(f"raw_edges_count: {edges_count}")    
    return G

def load_all_topics(data_directory):
    """
    批量读取所有主题文件，生成每个主题的图。
    :param data_directory: 数据文件所在目录
    :return: 主题图字典，key 是主题名称，value 是图结构
    """
    topic_graphs = {}
    for file_name in os.listdir(data_directory):
        if file_name.endswith(".net"):
            # 获取主题编号
            topic = os.path.splitext(file_name)[0]
            file_path = os.path.join(data_directory, file_name)
            # 读取单个主题的图
            topic_graph = read_citation_file(file_path)
            topic_graphs[f"{topic}"] = topic_graph
    
    return topic_graphs


def add_topic_vector_to_graph(graph, topic_index, num_topics):
    """
    为每个节点添加话题分布向量，特定主题权重最高。
    :param graph: 主题图
    :param topic_index: 当前图的主题索引
    :param num_topics: 总的话题数量
    """
    for node in graph.nodes():
        # 生成话题分布向量，特定主题权重最高
        topic_vector = np.full(num_topics, 0.0)  # 初始化为较小权重
        topic_vector[topic_index] = 1  # 将特定主题的权重设为较大值
        graph.nodes[node]['topics'] = topic_vector

def combine_topic_graphs(topic_graphs, base_graph):
    """
    合并多个主题图，使用 raw 数据作为底图，通过标题匹配节点。
    :param topic_graphs: 每个主题的图字典
    :param base_graph: 原始引文网络图（通过 read_citation_file_raw 读取）
    :return: 合并后的总图
    """
    # 创建标题到节点ID的映射
    title_to_id = {data['title']: node_id for node_id, data in base_graph.nodes(data=True)}
    
    # 为底图中的所有节点初始化话题分布向量
    num_topics = len(topic_graphs)
    for node in base_graph.nodes():
        base_graph.nodes[node]['topics'] = np.zeros(num_topics)
    
    # 遍历每个主题图，将话题分布添加到底图中匹配的节点
    for topic_graph in topic_graphs.values():
        for node, data in topic_graph.nodes(data=True):
            title = data['title']
            if title in title_to_id:
                base_id = title_to_id[title]
                # 累加话题分布
                base_graph.nodes[base_id]['topics'] += data['topics']
    
    # 归一化每个节点的话题分布向量
    for node in base_graph.nodes():
        topic_sum = base_graph.nodes[node]['topics'].sum()
        if topic_sum > 0:  # 避免除以零
            base_graph.nodes[node]['topics'] /= topic_sum
    
    return base_graph

def load_aminer_dataset(data_directory):
    # 加载底图（假设在同一目录下有一个 raw 数据文件）
    raw_file_path = os.path.join(data_directory, "../citation-raw.txt")
    raw_graph = read_citation_file_raw(raw_file_path)
    
    # 加载所有主题图
    topic_graphs = load_all_topics(data_directory)
    
    # 创建主题到索引的映射
    topic_to_index = {topic: idx for idx, topic in enumerate(sorted(topic_graphs.keys()))}
    
    # 为每个图添加主题向量，使用映射的索引
    for topic, topic_graph in topic_graphs.items():
        topic_index = topic_to_index[topic]
        add_topic_vector_to_graph(topic_graph, topic_index, len(topic_graphs))
    
    combined_graph = combine_topic_graphs(topic_graphs, raw_graph)
    return combined_graph, len(topic_graphs)

if __name__ == "__main__":
    data_directory = "./data/lfs.aminer.cn/graphs_pubs"
    combined_graph, num_topics = load_aminer_dataset(data_directory)
    
    print(f"总节点数: {combined_graph.number_of_nodes()}")
    print(f"总边数: {combined_graph.number_of_edges()}")

