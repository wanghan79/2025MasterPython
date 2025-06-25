#pragma once

#include "maxHeap.h"
#include "redBlackTree.h"
#include "hierarchicalClustering.h"
#include <unordered_set>
#include <map>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

struct parameters
{
    int k = 0;                                    // 簇的数量
    int numNodes = 0;                             // 节点数量
    int numEdges = 0;                             // 边的数量
    int tMin = 0;                                 // 最小扰动次数
    int tStep = 0;                                // 步长
    int tMax = 0;                                 // 最大扰动次数
    int limitTime = 0;                            // 限制时间
    string filename = "";                         // 文件名
    vector<double> nodeWeight;                    // 节点权重
    map<int, struct cluster *> hcResults;         // 层次聚类得到的簇结果
    map<int, RedBlackTree *> rbTreesResult;       // 层次聚类得到的红黑树结果
    vector<unordered_map<int, double> *> adjList; // 邻接表
};

class vns
{
public:
    struct parameters &params;                    // 参数结构体
    double bestNCutChange = 0;                    // 最佳NCut变化
    double tempNCutChange = 0;                    // 临时NCut变化
    int k = 0;                                    // 簇的数量
    int numNodes = 0;                             // 节点数量
    int numEdges = 0;                             // 边的数量
    int tMin = 0;                                 // 最小扰动次数
    int tMax = 0;                                 // 最大扰动次数
    int tStep = 0;                                // 步长
    int limitTime = 0;                            // 限制时间
    MaxHeap *mh;                                  // 最大堆
    vector<int> nodesID;                          // 节点ID
    vector<double> nodeWeight;                    // 每个簇的总权重
    map<int, int> label;                          // 节点标签，节点所属簇
    map<int, struct cluster *> clusters;          // 簇数组
    map<int, RedBlackTree *> rbTrees;             // 层次聚类得到的红黑树结果
    vector<unordered_map<int, double> *> adjList; // 邻接表
public:
    vns(struct parameters &params); // 构造函数
    void runVNS();
    void initParamers();                               // 初始化函数
    void initResults();                                // 初始化结
    void mergeClusters(int i, int j);                  // 合并簇
    void shaking(int t);                               // 扰动函数
    void swapNodes(int i, int j);                      // 交换节点
    void moveNode(int node, int cluster);              // 移动节点
    void fastLocalSearch();                            // 在邻居找到局部搜索
    void completedLocalSearch();                       // 在所有簇找到完成局部搜索
    void findMerge(int &clusteri, int &clusterj);      // 找到最好的合并
    void calculateSAllWeight();                        // 计算所有簇之间权重
    double calculateNCutChange(int node, int cluster); // 计算NCut变化
    double calculateNCut();                            // 计算NCut
    bool isneighbor(int nodei, int clusterj);          // 判断nodei是否和clusterj相邻
};
