from python.maxHeap import MaxHeap, MHNode
from python.red_black_tree import RedBlackTree, RBTNode, LinkListNode
import os

class Cluster:
    """簇的数据结构"""
    def __init__(self, idx):
        self.flag = 1                # 标志位,表示是否被删除
        self.id = idx                # 簇ID,数组中的索引
        self.inweight = 0.0          # 簇的内部权重
        self.outweight = 0.0         # 簇的外部权重
        self.totalWeight = 0.0       # 簇的总权重
        self.nodes = set()           # 包含的节点
        self.outWeight = {}          # 与其他簇的权重
        self.delta = {}              # 与其他簇合并后的delta矩阵

class HierarchicalClustering:
    """层次聚类算法类"""
    def __init__(self, percent=0.5):
        self.percent = percent        # 处理的百分比
        self.numNodes = 0             # 节点数量
        self.numClusters = 0          # 聚类数量
        self.numEdges = 0             # 边的数量
        self.isWeighted = 0           # 是否加权
        self.nodesStartFromOne = 0    # 节点是否从1开始
        self.mh = None                # 最大堆
        self.labels = []              # 节点标签
        self.clusters = []            # 簇数组
        self.rbTrees = []             # 红黑树数组
        self.totalweights = []        # 每个簇的总权重
        self.inWeights = []           # 每个簇的内部权重
        self.adjList = []             # 邻接表

    def readFile(self, filename):
        """读取图文件"""
        with open(filename, 'r') as f:
            line = f.readline()
            parts = line.strip().split()
            self.numNodes = int(parts[0])
            self.numEdges = int(parts[1])
            self.isWeighted = int(parts[2]) if len(parts) > 2 else 0

        # 初始化数据结构
        self.adjList = [None] * self.numNodes
        self.totalweights = [0.0] * self.numNodes
        self.inWeights = [0.0] * self.numNodes
        
        # 根据是否有权重选择不同的读取方法
        if self.isWeighted:
            self.readWeightedFile(filename)
        else:
            self.readUnweightedFile(filename)

    def readUnweightedFile(self, filename):
        """读取无权重图文件"""
        with open(filename, 'r') as f:
            f.readline()  # 跳过第一行
            nodeID = 0
            for line in f:
                pMap = {}
                for neighborID in map(int, line.strip().split()):
                    if neighborID < 1 or neighborID > self.numNodes:
                        print(f"Invalid neighbor ID: {neighborID}")
                        continue
                    neighborID -= 1  # 文件是从1开始的，这里从0开始
                    pMap[neighborID] = 1.0
                    self.totalweights[nodeID] += 1.0
                self.adjList[nodeID] = pMap
                nodeID += 1

    def readWeightedFile(self, filename):
        """读取有权重图文件"""
        with open(filename, 'r') as f:
            f.readline()  # 跳过第一行
            nodeID = 0
            for line in f:
                pMap = {}
                items = line.strip().split()
                i = 0
                while i < len(items) - 1:
                    neighborID = int(items[i])
                    weight = float(items[i + 1])
                    if neighborID < 1 or neighborID > self.numNodes:
                        print(f"Invalid neighbor ID: {neighborID}")
                        i += 2
                        continue
                    neighborID -= 1  # 文件是从1开始的，这里从0开始
                    pMap[neighborID] = weight
                    self.totalweights[nodeID] += weight
                    i += 2
                self.adjList[nodeID] = pMap
                nodeID += 1

    def init(self):
        """初始化红黑树和最大堆"""
        self.mh = MaxHeap()
        self.numClusters = self.numNodes
        
        # 初始化每个簇的信息
        for i in range(self.numNodes):
            c = Cluster(i)
            c.totalWeight = self.totalweights[i]
            c.nodes.add(i)
            
            rbTree = RedBlackTree()
            for j, wij in self.adjList[i].items():
                # 计算合并簇i和j的delta值
                nodeiweight = self.totalweights[i]
                nodejweight = self.totalweights[j]
                delta = (2 * wij) / (nodeiweight + nodejweight)
                
                # 生成最大堆节点
                mhnode = MHNode(i, j, delta)
                self.mh.insert(mhnode)
                
                # 生成红黑树节点
                rbtnode = RBTNode(neighbor=j, weight=wij, heap_ptr=mhnode)
                rbTree.insert(rbtnode)
                
            self.rbTrees.append(rbTree)
            self.clusters.append(c)

    def performClustering(self):
        """执行层次聚类"""
        while self.numClusters > self.numNodes * self.percent:
            # 反复取最大堆顶，直到遇到有效的节点
            maxNode = None
            while True:
                if self.mh.is_empty():
                    break
                maxNode = self.mh.get_max()
                # 检查两个簇是否都有效
                if self.clusters[maxNode.i].flag != 0 and self.clusters[maxNode.j].flag != 0:
                    break
                # 否则移除该节点，继续
                self.mh.remove_max()
            
            if not maxNode or self.mh.is_empty():
                break  # 堆空则退出
            
            i, j = maxNode.i, maxNode.j
            if i > j:
                i, j = j, i
            
            # 合并簇i和簇j
            self.mergeClusters(i, j)

    def mergeClusters(self, i, j):
        """合并簇i和簇j"""
        if not self.rbTrees[i] or not self.rbTrees[j]:
            return
        
        rbTreei = self.rbTrees[i]
        rbTreej = self.rbTrees[j]
        
        # 1. 合并簇j的所有邻居到簇i
        jList = rbTreej.get_link_list()
        curr = jList
        wij = 0  # 记录i-j边的权重
        
        while curr is not None:
            k = curr.rbtnode.neighbor
            wjk = curr.rbtnode.weight
            
            if k == i:
                # i和j是邻居，这条边变成内部边
                wij = wjk
                # 删除i红黑树中指向j的节点和最大堆节点
                ij = rbTreei.search_tree(j)
                if ij and ij.heap_ptr:
                    self.mh.delete_node(ij.heap_ptr.index)
                rbTreei.delete_node(j)
                
                # 删除j红黑树中指向i的节点和最大堆节点
                ji = rbTreej.search_tree(i)
                if ji and ji.heap_ptr:
                    self.mh.delete_node(ji.heap_ptr.index)
                rbTreej.delete_node(i)
                
                curr = curr.next
                continue
            
            # 查找i的红黑树中是否有k
            node_ik = rbTreei.search_tree(k)
            
            if node_ik:
                # i和j都与k相邻，合并权重
                node_ik.weight += wjk
                
                # 更新最大堆节点的delta
                mhNode = node_ik.heap_ptr
                wik = node_ik.weight
                wii = self.clusters[i].inweight
                wkk = self.clusters[k].inweight
                twi = self.clusters[i].totalWeight
                twk = self.clusters[k].totalWeight
                newDelta = (wii + wkk + 2 * wik) / (twi + twk) - wii / twi - wkk / twk
                mhNode.delta = newDelta
                self.mh.heapify_up(mhNode.index)
                self.mh.heapify_down(mhNode.index)
            else:
                # k只在j，不在i，插入新节点到i的红黑树和最大堆
                mhNode = MHNode(i, k, (2 * wjk) / (self.clusters[i].totalWeight + self.clusters[k].totalWeight))
                self.mh.insert(mhNode)
                
                rbtNode = RBTNode(neighbor=k, weight=wjk, heap_ptr=mhNode)
                rbTreei.insert(rbtNode)
            
            curr = curr.next
        
        # 1.5 删除所有红黑树和最大堆中neighbor为j的节点（除了i和j自己）
        for idx in range(len(self.rbTrees)):
            if self.rbTrees[idx] and idx != j and idx != i:
                # 删除 idx->j
                node = self.rbTrees[idx].search_tree(j)
                if node:
                    if node.heap_ptr:
                        self.mh.delete_node(node.heap_ptr.index)
                    self.rbTrees[idx].delete_node(j)
                
                # 删除 j->idx（对称处理，防止堆中有(j, idx)组合）
                if self.rbTrees[j]:
                    node2 = self.rbTrees[j].search_tree(idx)
                    if node2 and node2.heap_ptr:
                        self.mh.delete_node(node2.heap_ptr.index)
        
        # 2. 合并节点集合、权重
        self.clusters[i].nodes.update(self.clusters[j].nodes)
        self.clusters[i].totalWeight += self.clusters[j].totalWeight
        
        # 3. 更新inweight（i与j之间的边权*2，加上j的内部权重）
        self.clusters[i].inweight += self.clusters[j].inweight + 2 * wij
        
        # 4. 删除i到j的边（如已在上面处理）
        
        # 5. 删除簇j的红黑树
        rbTreej.destroy_tree(rbTreej.get_root())
        self.rbTrees[j] = None
        
        # 6. 标记j已被合并
        self.clusters[j].flag = 0
        self.numClusters -= 1
        
        # 7. 更新所有最大堆中与i相关的节点的delta，并删除所有与j相关的节点
        # 遍历时不能直接删除元素，需先收集待删除的索引
        toDelete = []
        for idx, node in enumerate(self.mh.heap):
            if node.i == j or node.j == j:
                # 标记与j相关的节点，待会统一删除
                toDelete.append(idx)
                continue
                
            if node.i == i or node.j == i:
                a, b = node.i, node.j
                wii = self.clusters[a].inweight
                wjj = self.clusters[b].inweight
                twi = self.clusters[a].totalWeight
                twj = self.clusters[b].totalWeight
                rbtnode = self.rbTrees[a].search_tree(b) if self.rbTrees[a] else None
                wij2 = rbtnode.weight if rbtnode else 0
                node.delta = (wii + wjj + 2 * wij2) / (twi + twj) - wii / twi - wjj / twj
                self.mh.heapify_up(node.index)
                self.mh.heapify_down(node.index)
        
        # 倒序删除，避免索引混乱
        for k in reversed(toDelete):
            self.mh.delete_node(k)

    def run(self, percent, filename):
        """运行层次聚类算法，返回聚类结果"""
        self.percent = percent
        self.readFile(filename)
        self.init()
        self.performClustering()
        
        # 返回聚类结果（每个有效簇的节点集合）
        clusters = []
        for c in self.clusters:
            if c.flag == 1:  # 只保留有效的簇
                clusters.append(list(c.nodes))
        
        return clusters

    def get_clusters(self):
        """获取当前的聚类结果"""
        result = []
        for c in self.clusters:
            if c.flag == 1:
                result.append(list(c.nodes))
        return result


# 测试代码
if __name__ == "__main__":
    # 示例使用
    hc = HierarchicalClustering()
    
    # 设置压缩比例和输入文件
    percent = 0.5
    filename = "instances/karate.graph"
    
    # 执行聚类
    clusters = hc.run(percent, filename)
    
    # 打印结果
    print(f"图压缩为 {len(clusters)} 个簇（原始节点数: {hc.numNodes}）")
    for i, cluster in enumerate(clusters):
        print(f"簇 {i+1}: {sorted(cluster)}")