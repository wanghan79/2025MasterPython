from python.maxHeap import MaxHeap, MHNode
from python.red_black_tree import RedBlackTree
from python.hcluster import Cluster, HierarchicalClustering
import random
import time
import sys
from collections import defaultdict

class Parameters:
    """参数结构体"""
    def __init__(self):
        self.k = 0                     # 簇的数量
        self.numNodes = 0              # 节点数量
        self.numEdges = 0              # 边的数量
        self.tMin = 0                  # 最小扰动次数
        self.tStep = 0                 # 步长
        self.tMax = 0                  # 最大扰动次数
        self.limitTime = 0             # 限制时间
        self.filename = ""             # 文件名
        self.nodeWeight = []           # 节点权重
        self.hcResults = {}            # 层次聚类得到的簇结果
        self.rbTreesResult = {}        # 层次聚类得到的红黑树结果
        self.adjList = []              # 邻接表


class VNS:
    """变邻域搜索算法类"""
    def __init__(self, params):
        """构造函数"""
        self.params = params
        self.bestNCutChange = 0.0        # 最佳NCut变化
        self.tempNCutChange = 0.0        # 临时NCut变化
        self.k = 0                       # 簇的数量
        self.numNodes = 0                # 节点数量
        self.numEdges = 0                # 边的数量
        self.tMin = 0                    # 最小扰动次数
        self.tMax = 0                    # 最大扰动次数
        self.tStep = 0                   # 步长
        self.limitTime = 0               # 限制时间
        self.mh = None                   # 最大堆
        self.nodesID = []                # 节点ID
        self.nodeWeight = []             # 每个簇的总权重
        self.label = {}                  # 节点标签，记录节点所属簇
        self.clusters = {}               # 簇字典
        self.rbTrees = {}                # 层次聚类得到的红黑树结果
        self.adjList = []                # 邻接表

    def init_parameters(self):
        """初始化参数"""
        self.k = self.params.k
        self.numNodes = self.params.numNodes
        self.numEdges = self.params.numEdges
        self.tMin = self.params.tMin
        self.tMax = self.params.tMax
        self.tStep = self.params.tStep
        self.nodeWeight = self.params.nodeWeight  # 初始化节点权重
        self.limitTime = self.params.limitTime    # 初始化限制时间
        
        # 初始化簇并移除无效簇
        for cluster_id, cluster in self.params.hcResults.items():
            if cluster is not None and cluster.flag == 1:
                self.clusters[cluster_id] = cluster
                # 为每个节点初始化标签（所属簇）
                for node in cluster.nodes:
                    self.label[node] = cluster_id
        
        # 初始化邻接表
        self.adjList = self.params.adjList
        
        # 初始化红黑树
        for tree_id, tree in self.params.rbTreesResult.items():
            if tree_id in self.clusters:
                self.rbTrees[tree_id] = tree

    def init_results(self):
        """初始化解，将簇数量缩减到k"""
        size = len(self.clusters)
        while size > self.k:
            # 找到两个要合并的簇
            cluster_i, cluster_j = self.find_merge()
            if cluster_i is None or cluster_j is None:
                print("无法找到更多可合并的簇")
                break  # 如果找不到两个可合并的簇，则退出循环
            
            # 合并簇
            self.merge_clusters(cluster_i, cluster_j)
            
            # 更新簇的数量
            size = len(self.clusters)
        
        self.calculate_all_weight()  # 计算所有簇之间的权重
        
        # 初始化节点ID数组
        self.nodesID = list(range(self.numNodes))

    def run_vns(self):
        """运行VNS算法"""
        self.init_parameters()
        self.init_results()

        t = self.tMin
        start_time = time.time()
        duration = 0

        current_ncut = self.calculate_ncut()  # 当前ncut
        last_ncut = 999.0                     # 上一次ncut
        improve_ncut_change = 999.0           # 改进的ncut变化
        
        random.seed(time.time())
        
        while duration < self.limitTime:
            if random.randint(0, 99) > 99:
                # 在所有可能位置进行搜索
                self.completed_local_search()
            else:
                # 在邻居中进行搜索
                self.fast_local_search()
                
            last_ncut = current_ncut  # 保存上一次ncut
            current_ncut = self.calculate_ncut()
            improve_ncut_change = last_ncut - current_ncut  # 计算改进的ncut变化
            
            if self.tempNCutChange <= self.bestNCutChange and improve_ncut_change >= 0.0001:
                # 如果找到更好的解，接受并增加扰动次数
                self.bestNCutChange = self.tempNCutChange
                t += self.tStep
            else:
                # 如果没有找到更好的解，减小扰动
                t -= self.tStep
                if t < self.tMin:
                    t = self.tMin
            
            if t >= self.tMax:
                t = self.tMin  # 重置扰动次数
                
            self.shaking(t)  # 扰动
            
            duration = time.time() - start_time
            print(f"ncut: {current_ncut:.6f}    delta: {improve_ncut_change:.6f}    t: {t}    time: {int(duration)}")

    def merge_clusters(self, i, j):
        """合并簇i和簇j"""
        # 确保簇i和簇j都存在
        if i not in self.clusters or j not in self.clusters:
            print(f"警告：尝试合并不存在的簇，i={i}, j={j}")
            return
        
        # 更新被合并簇j中所有节点的label为i
        for node in self.clusters[j].nodes:
            self.label[node] = i
        
        # 合并节点集合
        self.clusters[i].nodes.update(self.clusters[j].nodes)
        
        # 更新簇i的总权重
        self.clusters[i].totalWeight += self.clusters[j].totalWeight
        
        # 更新簇i的内部权重（加上j的内部权重，以及i和j之间的边权重*2）
        # 计算i和j之间的边权重
        wij = 0.0
        for node_i in self.clusters[i].nodes:
            if node_i in self.clusters[j].nodes:
                continue  # 跳过j中的节点，因为还没删除
            for node_j in self.clusters[j].nodes:
                if node_j in self.adjList[node_i]:
                    wij += self.adjList[node_i][node_j]
        
        self.clusters[i].inweight += self.clusters[j].inweight + 2 * wij
        
        # 删除簇j
        del self.clusters[j]

    def shaking(self, t):
        """随机扰动t次"""
        random.seed(time.time())  # 设置随机种子
        
        # 获取簇的列表
        keys = list(self.clusters.keys())
        
        # 随机选择两个簇进行交换t次
        for _ in range(t):
            if len(keys) < 2:
                break
                
            idx1 = random.randint(0, len(keys) - 1)
            idx2 = random.randint(0, len(keys) - 2)
            
            # 保证idx1和idx2不同
            if idx2 >= idx1:
                idx2 += 1
                
            cluster_i = keys[idx1]
            cluster_j = keys[idx2]
            
            # 随机选择簇中的节点
            if not self.clusters[cluster_i].nodes or not self.clusters[cluster_j].nodes:
                continue
                
            node_i = random.choice(list(self.clusters[cluster_i].nodes))
            node_j = random.choice(list(self.clusters[cluster_j].nodes))
            
            # 交换节点
            self.swap_nodes(node_i, node_j)

    def fast_local_search(self):
        """在邻居中进行局部搜索"""
        flag = True
        while flag:
            flag = False
            
            # 尝试对每个节点进行移动
            for node in range(self.numNodes):
                if node not in self.label:
                    continue
                    
                current_cluster = self.label[node]
                best_ncut_change = 0.0
                best_cluster = current_cluster
                
                # 查找当前节点的邻居节点所在的簇
                neighbor_clusters = set()
                if node < len(self.adjList) and self.adjList[node]:
                    for neighbor, weight in self.adjList[node].items():
                        if neighbor in self.label:
                            neighbor_cluster = self.label[neighbor]
                            if neighbor_cluster != current_cluster:
                                neighbor_clusters.add(neighbor_cluster)
                
                # 尝试将节点移动到邻居簇
                for cluster in neighbor_clusters:
                    ncut_change = self.calculate_ncut_change(node, cluster)
                    if ncut_change < best_ncut_change:
                        best_ncut_change = ncut_change
                        best_cluster = cluster
                
                # 如果找到更好的簇，进行移动
                if best_cluster != current_cluster:
                    self.move_node(node, best_cluster)
                    self.tempNCutChange = best_ncut_change
                    flag = True
                    break  # 找到更好的移动后，立即重新开始搜索

    def completed_local_search(self):
        """在所有簇中进行完整局部搜索"""
        flag = True
        while flag:
            flag = False
            
            # 尝试对每个节点进行移动
            for node in range(self.numNodes):
                if node not in self.label:
                    continue
                    
                current_cluster = self.label[node]
                best_ncut_change = 0.0
                best_cluster = current_cluster
                
                # 尝试将节点移动到所有其他簇
                for cluster in self.clusters.keys():
                    if cluster != current_cluster:
                        ncut_change = self.calculate_ncut_change(node, cluster)
                        if ncut_change < best_ncut_change:
                            best_ncut_change = ncut_change
                            best_cluster = cluster
                
                # 如果找到更好的簇，进行移动
                if best_cluster != current_cluster:
                    self.move_node(node, best_cluster)
                    self.tempNCutChange = best_ncut_change
                    flag = True
                    break  # 找到更好的移动后，立即重新开始搜索

    def calculate_ncut_change(self, node, cluster):
        """计算将节点node移动到簇cluster的NCut变化"""
        if node not in self.label or cluster not in self.clusters:
            return float('inf')
            
        ci = self.label[node]         # 节点所属的簇
        cj = cluster                  # 目标簇
        
        if ci not in self.clusters:
            return float('inf')
        
        # 获取簇的相关权重
        owi = self.clusters[ci].outweight   # 簇 i 的外部权重
        owj = self.clusters[cj].outweight   # 簇 j 的外部权重
        iwi = self.clusters[ci].inweight    # 簇 i 的内部权重
        iwj = self.clusters[cj].inweight    # 簇 j 的内部权重
        twi = self.clusters[ci].totalWeight # 簇 i 的总权重
        twj = self.clusters[cj].totalWeight # 簇 j 的总权重
        
        if twi <= 0 or twj <= 0:
            return float('inf')
        
        # 计算旧的cut值
        old_cut = owi / twi + owj / twj
        
        # 计算节点在簇i中的内部权重
        in_weight_i = 0.0
        for n in self.clusters[ci].nodes:
            if n != node and node < len(self.adjList) and self.adjList[node] and n in self.adjList[node]:
                in_weight_i += self.adjList[node][n]
        
        # 计算i移除节点后的内部权重
        iwi_new = iwi - 2 * in_weight_i
        # 计算i移除节点后的总权重
        twi_new = twi - self.nodeWeight[node]
        # 计算i移除节点后的外部权重
        owi_new = twi_new - iwi_new
        
        # 计算节点在簇j中的内部权重
        in_weight_j = 0.0
        for n in self.clusters[cj].nodes:
            if node < len(self.adjList) and self.adjList[node] and n in self.adjList[node]:
                in_weight_j += self.adjList[node][n]
        
        # 计算j添加节点后的内部权重
        iwj_new = iwj + 2 * in_weight_j
        # 计算j添加节点后的总权重
        twj_new = twj + self.nodeWeight[node]
        # 计算j添加节点后的外部权重
        owj_new = twj_new - iwj_new
        
        if twi_new <= 0 or twj_new <= 0:
            return float('inf')
        
        # 计算新的cut值
        new_cut = owi_new / twi_new + owj_new / twj_new
        
        # 计算cut值变化
        cut_change = new_cut - old_cut
        return cut_change

    def find_merge(self):
        """找到要合并的两个簇（当前实现是随机选择）"""
        keys = list(self.clusters.keys())
        if len(keys) < 2:
            return None, None  # 返回None表示无法找到两个簇来合并
        
        # 随机选择两个不同的簇
        idx1 = random.randint(0, len(keys) - 1)
        idx2 = random.randint(0, len(keys) - 2)
        
        # 保证idx1和idx2不同
        if idx2 >= idx1:
            idx2 += 1
        
        cluster_i = keys[idx1]
        cluster_j = keys[idx2]
        
        return cluster_i, cluster_j  # 返回两个实际存在的簇ID

    def calculate_all_weight(self):
        """计算所有簇的权重"""
        # 计算节点权重（如果尚未计算）
        if not self.nodeWeight or len(self.nodeWeight) != self.numNodes:
            self.nodeWeight = [0.0] * self.numNodes
            for node_id in range(self.numNodes):
                if node_id < len(self.adjList) and self.adjList[node_id]:
                    for neighbor, weight in self.adjList[node_id].items():
                        self.nodeWeight[node_id] += weight
        
        # 计算簇的总权重
        for cluster_id, cluster in self.clusters.items():
            total_weight = 0.0
            for node in cluster.nodes:
                if node < len(self.nodeWeight):
                    total_weight += self.nodeWeight[node]
            cluster.totalWeight = total_weight
        
        # 计算簇的内部权重和外部权重
        for cluster_id, cluster in self.clusters.items():
            in_weight = 0.0
            for node_i in cluster.nodes:
                for node_j in cluster.nodes:
                    if node_i < len(self.adjList) and self.adjList[node_i] and node_j in self.adjList[node_i]:
                        in_weight += self.adjList[node_i][node_j]
            
            cluster.inweight = in_weight
            cluster.outweight = cluster.totalWeight - cluster.inweight

    def calculate_ncut(self):
        """计算当前解的归一化割集（NCut）值"""
        ncut = 0.0
        for cluster in self.clusters.values():
            if cluster.totalWeight > 0:
                ncut += cluster.outweight / cluster.totalWeight
        
        return ncut

    def move_node(self, node, cluster):
        """将节点node移动到簇cluster"""
        if node not in self.label or cluster not in self.clusters:
            return
            
        node_i = node
        cluster_i = self.label[node_i]
        cluster_j = cluster
        
        if cluster_i not in self.clusters:
            return
        
        # 计算节点在原簇中的内部权重
        in_weight_node_i = 0.0
        for n in self.clusters[cluster_i].nodes:
            if n != node_i and node_i < len(self.adjList) and self.adjList[node_i] and n in self.adjList[node_i]:
                in_weight_node_i += self.adjList[node_i][n]
        
        # 计算节点在新簇中的内部权重
        in_weight_node_j = 0.0
        for n in self.clusters[cluster_j].nodes:
            if node_i < len(self.adjList) and self.adjList[node_i] and n in self.adjList[node_i]:
                in_weight_node_j += self.adjList[node_i][n]
        
        # 更新簇的内部权重
        self.clusters[cluster_i].inweight -= in_weight_node_i * 2
        self.clusters[cluster_j].inweight += in_weight_node_j * 2
        
        # 更新簇的总权重
        if node_i < len(self.nodeWeight):
            self.clusters[cluster_i].totalWeight -= self.nodeWeight[node_i]
            self.clusters[cluster_j].totalWeight += self.nodeWeight[node_i]
        
        # 更新簇的外部权重
        self.clusters[cluster_i].outweight = self.clusters[cluster_i].totalWeight - self.clusters[cluster_i].inweight
        self.clusters[cluster_j].outweight = self.clusters[cluster_j].totalWeight - self.clusters[cluster_j].inweight
        
        # 更新簇的节点集合
        self.clusters[cluster_i].nodes.remove(node_i)
        self.clusters[cluster_j].nodes.add(node_i)
        
        # 更新节点标签
        self.label[node_i] = cluster_j

    def swap_nodes(self, i, j):
        """交换两个节点的簇"""
        if i not in self.label or j not in self.label:
            return
            
        node_i = i
        node_j = j
        cluster_i = self.label[node_i]
        cluster_j = self.label[node_j]
        
        if cluster_i not in self.clusters or cluster_j not in self.clusters:
            return
        
        # 修改节点集合
        self.clusters[cluster_i].nodes.remove(node_i)
        self.clusters[cluster_j].nodes.remove(node_j)
        
        # 计算节点i在簇i中的内部权重
        in_weight_node_i_in_i = 0.0
        for node in self.clusters[cluster_i].nodes:
            if node_i < len(self.adjList) and self.adjList[node_i] and node in self.adjList[node_i]:
                in_weight_node_i_in_i += self.adjList[node_i][node]
        
        # 更新簇i的内部权重
        self.clusters[cluster_i].inweight -= in_weight_node_i_in_i * 2
        
        # 计算节点j在簇j中的内部权重
        in_weight_node_j_in_j = 0.0
        for node in self.clusters[cluster_j].nodes:
            if node_j < len(self.adjList) and self.adjList[node_j] and node in self.adjList[node_j]:
                in_weight_node_j_in_j += self.adjList[node_j][node]
        
        # 更新簇j的内部权重
        self.clusters[cluster_j].inweight -= in_weight_node_j_in_j * 2
        
        # 计算节点i在簇j中的内部权重
        in_weight_node_i_in_j = 0.0
        for node in self.clusters[cluster_j].nodes:
            if node_i < len(self.adjList) and self.adjList[node_i] and node in self.adjList[node_i]:
                in_weight_node_i_in_j += self.adjList[node_i][node]
        
        # 更新簇j加入节点i后的内部权重
        self.clusters[cluster_j].inweight += in_weight_node_i_in_j * 2
        
        # 计算节点j在簇i中的内部权重
        in_weight_node_j_in_i = 0.0
        for node in self.clusters[cluster_i].nodes:
            if node_j < len(self.adjList) and self.adjList[node_j] and node in self.adjList[node_j]:
                in_weight_node_j_in_i += self.adjList[node_j][node]
        
        # 更新簇i加入节点j后的内部权重
        self.clusters[cluster_i].inweight += in_weight_node_j_in_i * 2
        
        # 计算总权重变化
        if node_i < len(self.nodeWeight) and node_j < len(self.nodeWeight):
            self.clusters[cluster_i].totalWeight -= self.nodeWeight[node_i]
            self.clusters[cluster_i].totalWeight += self.nodeWeight[node_j]
            self.clusters[cluster_j].totalWeight -= self.nodeWeight[node_j]
            self.clusters[cluster_j].totalWeight += self.nodeWeight[node_i]
        
        # 更新簇的外部权重
        self.clusters[cluster_i].outweight = self.clusters[cluster_i].totalWeight - self.clusters[cluster_i].inweight
        self.clusters[cluster_j].outweight = self.clusters[cluster_j].totalWeight - self.clusters[cluster_j].inweight
        
        # 更新簇的节点集合和标签
        self.clusters[cluster_i].nodes.add(node_j)
        self.clusters[cluster_j].nodes.add(node_i)
        self.label[node_i] = cluster_j
        self.label[node_j] = cluster_i

    def is_neighbor(self, node_i, cluster_j):
        """判断节点node_i是否和簇cluster_j相邻"""
        if cluster_j not in self.clusters or node_i >= len(self.adjList) or not self.adjList[node_i]:
            return False
            
        for node in self.clusters[cluster_j].nodes:
            if node in self.adjList[node_i]:
                return True
        return False


# 测试代码
if __name__ == "__main__":
    # 示例使用
    from python.hcluster import HierarchicalClustering
    
    # 创建层次聚类实例
    hc = HierarchicalClustering()
    
    # 设置压缩比例和输入文件
    percent = 0.5
    filename = "instances/karate.graph"
    
    # 执行层次聚类
    hc.readFile(filename)
    hc.init()
    hc.performClustering()
    
    # 设置VNS参数
    params = Parameters()
    params.k = 3  # 目标簇数
    params.numNodes = hc.numNodes
    params.numEdges = hc.numEdges
    params.tMin = 1
    params.tStep = 1
    params.tMax = 5
    params.limitTime = 10  # 10秒限制时间（减少测试时间）
    params.filename = filename
    params.nodeWeight = [0.0] * hc.numNodes
    
    # 计算节点权重
    for i in range(hc.numNodes):
        if hc.adjList[i]:
            for j, weight in hc.adjList[i].items():
                params.nodeWeight[i] += weight
    
    # 设置层次聚类结果 - 修正这里的逻辑
    valid_clusters = {}
    for i, cluster in enumerate(hc.clusters):
        if cluster is not None and cluster.flag == 1:
            valid_clusters[i] = cluster
    
    params.hcResults = valid_clusters
    params.rbTreesResult = {i: tree for i, tree in enumerate(hc.rbTrees) if tree is not None}
    params.adjList = hc.adjList
    
    print(f"层次聚类结果：{len(valid_clusters)} 个有效簇")
    for cid, cluster in valid_clusters.items():
        print(f"簇 {cid}: {sorted(list(cluster.nodes))}")
    
    # 创建VNS实例
    vns = VNS(params)
    
    # 运行VNS算法
    vns.run_vns()
    
    # 打印结果
    clusters = [(cid, sorted(list(cluster.nodes))) for cid, cluster in vns.clusters.items()]
    print("\nVNS聚类结果:")
    for cid, nodes in sorted(clusters):
        print(f"簇 {cid}: {nodes}")
    
    print(f"\n最终NCut值: {vns.calculate_ncut():.6f}")