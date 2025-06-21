#pragma once
#include "maxHeap.h"
#include "redBlackTree.h"
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
using namespace std;

struct cluster
{
    int flag = 1;               // 标志位,表示是否被删除
    int id = 0;                 // 簇ID,数组中的索引
    double inweight = 0.0;      // 簇的内部权重
    double outweight = 0.0;     // 簇的外部权重
    double totalWeight = 0.0;   // 簇的总权重
    unordered_set<int> nodes;   // 包含的节点
    map<int, double> outWeight; // 与其他簇的权重
    map<int, double> delta;     // 与其他簇合并后的delta矩阵
};

class HierarchicalClustering
{

public:
    HierarchicalClustering();                        // 构造函数
    void run(double &percent, string filename);      // 构造函数
    void readFile(const string &filename);           // 读取文件
    void readUnweightedFile(const string &filename); // 读取无权重文件
    void readWeightedFile(const string &filename);   // 读取有权重文件
    void init();                                     // 初始化红黑树和最大堆
    void performClustering();                        // 执行层次聚类
    void mergeClusters(int i, int j);                // 合并两个簇

public:
    double &percent;                              // 处理的百分比
    int numNodes;                                 // 节点数量
    int numClusters;                              // 聚类数量
    int numEdges;                                 // 边的数量
    int isWeighted;                               // 是否加权
    int nodesStartFromOne;                        // 节点是否从1开始
    MaxHeap *mh;                                  // 最大堆
    vector<int> labels;                           // 节点标签
    vector<struct cluster *> clusters;            // 簇数组
    vector<RedBlackTree *> rbTrees;               // 红黑树数组
    vector<double> totalweights;                  // 每个簇的总权重
    vector<double> inWeights;                     // 每个簇的内部权重
    vector<unordered_map<int, double> *> adjList; // 邻接表
};