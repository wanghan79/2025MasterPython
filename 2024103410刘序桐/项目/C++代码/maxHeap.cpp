#include "maxHeap.h"

// 获取父节点索引
int MaxHeap::parent(int index)
{
    return (index - 1) / 2;
}

// 获取左子节点索引
int MaxHeap::leftChild(int index)
{
    return 2 * index + 1;
}

// 获取右子节点索引
int MaxHeap::rightChild(int index)
{
    return 2 * index + 2;
}

// 堆化向上
void MaxHeap::heapifyUp(int index)
{
    while (index > 0 && heap[parent(index)]->delta < heap[index]->delta)
    {
        std::swap(heap[parent(index)], heap[index]);
        heap[parent(index)]->index = parent(index); // 更新父节点的索引
        heap[index]->index = index;                 // 更新当前节点的索引
        index = parent(index);
    }
    heap[index]->index = index; // 确保最终位置的索引正确
}

// 堆化向下
void MaxHeap::heapifyDown(int index)
{
    int largest = index;
    int left = leftChild(index);
    int right = rightChild(index);

    if (left < heap.size() && heap[left]->delta > heap[largest]->delta)
    {
        largest = left;
    }

    if (right < heap.size() && heap[right]->delta > heap[largest]->delta)
    {
        largest = right;
    }

    if (largest != index)
    {
        std::swap(heap[index], heap[largest]);
        heap[index]->index = index;     // 更新当前节点的索引
        heap[largest]->index = largest; // 更新交换后节点的索引
        heapifyDown(largest);
    }
}

// 插入元素
void MaxHeap::insert(MHNode *node)
{
    heap.push_back(node);
    node->index = heap.size() - 1; // 设置新插入节点的索引
    heapifyUp(heap.size() - 1);
}

// 获取堆顶元素
MHNode *MaxHeap::getMax()
{
    if (heap.empty())
    {
        throw std::runtime_error("Heap is empty");
    }
    return heap[0];
}

// 删除堆顶元素
void MaxHeap::removeMax()
{
    if (heap.empty())
    {
        throw std::runtime_error("Heap is empty");
    }
    heap[0] = heap.back(); // 将最后一个元素放到堆顶
    heap[0]->index = 0;    // 更新堆顶元素的新索引
    heap.pop_back();
    if (!heap.empty())
    {
        heapifyDown(0);
    }
}

// 获取堆的大小
int MaxHeap::size() const
{
    return heap.size();
}

// 检查堆是否为空
bool MaxHeap::isEmpty() const
{
    return heap.empty();
}

// 根据索引删除元素
void MaxHeap::removeAtIndex(int index)
{
    if (index < 0 || index >= heap.size())
    {
        throw std::runtime_error("Index out of bounds");
    }

    // 将要删除的元素与最后一个元素交换
    std::swap(heap[index], heap.back());
    heap[index]->index = index; // 更新交换后节点的索引
    heap.pop_back();            // 删除最后一个元素

    // 恢复堆的性质
    if (index < heap.size())
    {
        heapifyDown(index); // 尝试向下堆化
        heapifyUp(index);   // 如果向下堆化无效，尝试向上堆化
    }
}

// 输出整个最大堆的内容
void MaxHeap::printHeap() const
{
    // std::cout << "MaxHeap contents:" << std::endl;
    for (const auto &node : heap)
    {
        /*std::cout << "Node(i: " << node->i
                  << ", j: " << node->j
                  << ", index: " << node->index
                  << ", delta: " << node->delta
                  << ")" << std::endl;*/
        std::cout << node->delta << " ";
    }
    std::cout << std::endl;
}

void MaxHeap::deleteNode(int index)
{
    // 根据索引删除元素
    // 将索引和最后一个元素互换
    // 然后删除最后一个元素,然后将换完的元素向下堆化
    // 如果向下堆化无效，尝试向上堆化
    if (index < 0 || index >= heap.size())
    {
        throw std::runtime_error("Index out of bounds");
    }
    std::swap(heap[index], heap.back());
    heap[index]->index = index; // 更新交换后节点的索引
    heap.pop_back();            // 删除最后一个元素
    if (index < heap.size())
    {
        heapifyDown(index); // 尝试向下堆化
        heapifyUp(index);   // 如果向下堆化无效，尝试向上堆化
    }

    // std::cout << "delte node success!" << std::endl;
}