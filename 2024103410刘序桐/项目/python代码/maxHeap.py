class MHNode:
    """最大堆节点类"""
    def __init__(self, i=0, j=0, delta=0.0):
        self.i = i          # 合并的簇id
        self.j = j          # 合并的簇id
        self.index = 0      # 该簇在最大堆中的索引
        self.delta = delta  # 最大堆排序依据


class MaxHeap:
    """最大堆类"""
    
    def __init__(self):
        self.heap = []  # 动态数组，存储MHNode对象
    
    def parent(self, index):
        """获取父节点索引"""
        return (index - 1) // 2
    
    def left_child(self, index):
        """获取左子节点索引"""
        return 2 * index + 1
    
    def right_child(self, index):
        """获取右子节点索引"""
        return 2 * index + 2
    
    def heapify_up(self, index):
        """堆化向上"""
        while index > 0 and self.heap[self.parent(index)].delta < self.heap[index].delta:
            parent_idx = self.parent(index)
            # 交换节点
            self.heap[parent_idx], self.heap[index] = self.heap[index], self.heap[parent_idx]
            # 更新索引
            self.heap[parent_idx].index = parent_idx
            self.heap[index].index = index
            index = parent_idx
        self.heap[index].index = index  # 确保最终位置的索引正确
    
    def heapify_down(self, index):
        """堆化向下"""
        largest = index
        left = self.left_child(index)
        right = self.right_child(index)
        
        if left < len(self.heap) and self.heap[left].delta > self.heap[largest].delta:
            largest = left
        
        if right < len(self.heap) and self.heap[right].delta > self.heap[largest].delta:
            largest = right
        
        if largest != index:
            # 交换节点
            self.heap[index], self.heap[largest] = self.heap[largest], self.heap[index]
            # 更新索引
            self.heap[index].index = index
            self.heap[largest].index = largest
            self.heapify_down(largest)
    
    def insert(self, node):
        """插入元素"""
        self.heap.append(node)
        node.index = len(self.heap) - 1  # 设置新插入节点的索引
        self.heapify_up(len(self.heap) - 1)
    
    def get_max(self):
        """获取堆顶元素"""
        if not self.heap:
            raise RuntimeError("Heap is empty")
        return self.heap[0]
    
    def remove_max(self):
        """删除堆顶元素"""
        if not self.heap:
            raise RuntimeError("Heap is empty")
        
        # 将最后一个元素放到堆顶
        self.heap[0] = self.heap[-1]
        self.heap[0].index = 0  # 更新堆顶元素的新索引
        self.heap.pop()
        
        if self.heap:
            self.heapify_down(0)
    
    def size(self):
        """获取堆的大小"""
        return len(self.heap)
    
    def is_empty(self):
        """检查堆是否为空"""
        return len(self.heap) == 0
    
    def remove_at_index(self, index):
        """根据索引删除元素"""
        if index < 0 or index >= len(self.heap):
            raise RuntimeError("Index out of bounds")
        
        # 将要删除的元素与最后一个元素交换
        self.heap[index], self.heap[-1] = self.heap[-1], self.heap[index]
        self.heap[index].index = index  # 更新交换后节点的索引
        self.heap.pop()  # 删除最后一个元素
        
        # 恢复堆的性质
        if index < len(self.heap):
            self.heapify_down(index)  # 尝试向下堆化
            self.heapify_up(index)    # 如果向下堆化无效，尝试向上堆化
    
    def delete_node(self, index):
        """删除指定节点，并更新树结构"""
        if index < 0 or index >= len(self.heap):
            raise RuntimeError("Index out of bounds")
        
        # 将索引和最后一个元素互换
        self.heap[index], self.heap[-1] = self.heap[-1], self.heap[index]
        self.heap[index].index = index  # 更新交换后节点的索引
        self.heap.pop()  # 删除最后一个元素
        
        if index < len(self.heap):
            self.heapify_down(index)  # 尝试向下堆化
            self.heapify_up(index)    # 如果向下堆化无效，尝试向上堆化
    
    def print_heap(self):
        """输出整个最大堆的内容"""
        for node in self.heap:
            print(f"{node.delta} ", end="")
        print()


# 示例使用
if __name__ == "__main__":
    # 创建最大堆
    max_heap = MaxHeap()
    
    # 创建一些节点并插入
    node1 = MHNode(1, 2, 10.5)
    node2 = MHNode(3, 4, 15.2)
    node3 = MHNode(5, 6, 8.7)
    node4 = MHNode(7, 8, 20.1)
    
    max_heap.insert(node1)
    max_heap.insert(node2)
    max_heap.insert(node3)
    max_heap.insert(node4)
    
    print("堆内容:")
    max_heap.print_heap()
    
    print(f"堆大小: {max_heap.size()}")
    print(f"最大元素的delta值: {max_heap.get_max().delta}")
    
    # 删除最大元素
    max_heap.remove_max()
    print("删除最大元素后:")
    max_heap.print_heap()
