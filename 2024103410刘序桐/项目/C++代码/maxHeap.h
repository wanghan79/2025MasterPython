#pragma once
#include <vector>
#include <iostream>

struct MHNode
{
    int i;        // 合并的簇id
    int j;        // 合并的簇id
    int index;    // 该簇在最大堆中的索引
    double delta; // 最大堆排序依据
};

// 最大堆类
class MaxHeap
{
public:
    std::vector<MHNode *> heap; // 动态数组，存储指向结构体的指针

    int parent(int index);       // 获取父节点索引
    int leftChild(int index);    // 获取左子节点索引
    int rightChild(int index);   // 获取右子节点索引
    void heapifyUp(int index);   // 堆化向上
    void heapifyDown(int index); // 堆化向下
public:
    void insert(MHNode *node);     // 插入元素
    MHNode *getMax();              // 获取堆顶元素
    void removeMax();              // 删除堆顶元素
    int size() const;              // 获取堆的大小
    bool isEmpty() const;          // 检查堆是否为空
    void removeAtIndex(int index); // 根据索引删除元素
    void deleteNode(int index);    // 删除指定节点,并更新树结构
    void printHeap() const;        // 输出整个最大堆的内容
};