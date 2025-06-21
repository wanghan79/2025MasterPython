#include "hierarchicalClustering.h"
HierarchicalClustering::HierarchicalClustering()
    : percent(*(new double(0.5))), numNodes(0), numClusters(0), numEdges(0),
      isWeighted(0), nodesStartFromOne(0), mh(nullptr)
{
    // 这里可以根据需要初始化成员变量
}
void HierarchicalClustering::readFile(const string &filename)
{
    // 这里只是初始读取文件,并不初始化红黑树和最大堆
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    string line;
    getline(file, line); // 读取第一行
    istringstream iss(line);
    iss >> numNodes >> numEdges >> isWeighted;
    file.close();

    adjList.resize(numNodes);           // 邻接表初始化
    totalweights.assign(numNodes, 0.0); // 每个簇的总权重初始化为0.0
    inWeights.assign(numNodes, 0.0);    // 每个簇的内部权重初始化为0.0
    if (isWeighted)
    {
        readWeightedFile(filename);
    }
    else
    {
        readUnweightedFile(filename);
    }
}

void HierarchicalClustering::readUnweightedFile(const string &filename)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    string line;
    int nodeID = 0;
    getline(file, line); // 跳过第一行
    while (getline(file, line))
    {
        int neighborID = -1;
        stringstream iss(line);
        unordered_map<int, double> *pMap = new unordered_map<int, double>(); // 创建邻接表的元素
        while (iss >> neighborID)
        {
            if (neighborID < 1 || neighborID > numNodes)
            {
                cerr << "Invalid neighbor ID: " << neighborID << endl;
                continue;
            }
            // 处理邻居ID
            // 这里可以添加代码来存储邻居信息
            neighborID--; // 文件是从1开始的这里从0开始
            pMap->insert({neighborID, 1.0});
            totalweights[nodeID] += 1.0; // 累加总权重
        }
        adjList[nodeID] = pMap; // 将邻接表存储到数组中
        nodeID++;
    }
    file.close();
}

void HierarchicalClustering::readWeightedFile(const string &filename)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    string line;
    int nodeID = 0;
    getline(file, line); // 跳过第一行
    while (getline(file, line))
    {
        int neighborID = -1;
        double weight = 0.0;
        stringstream iss(line);
        unordered_map<int, double> *pMap = new unordered_map<int, double>(); // 创建邻接表的元素
        while (iss >> neighborID >> weight)
        {
            if (neighborID < 1 || neighborID > numNodes)
            {
                cerr << "Invalid neighbor ID: " << neighborID << endl;
                continue;
            }
            // 处理邻居ID
            // 这里可以添加代码来存储邻居信息
            neighborID--; // 文件是从1开始的这里从0开始
            pMap->insert({neighborID, weight});
            totalweights[nodeID] += weight; // 累加总权重
        }
        adjList[nodeID] = pMap; // 将邻接表存储到数组中
        nodeID++;
    }
    file.close();
}

void HierarchicalClustering::init()
{

    mh = new MaxHeap();     // 创建最大堆
    numClusters = numNodes; // 初始化簇的数量为节点的数量
    // 初始化每个簇的信息
    for (int i = 0; i < numNodes; i++)
    {
        struct cluster *c = new struct cluster;
        c->flag = 1; // 标志位，表示簇有效
        c->id = i;
        c->inweight = 0.0;
        c->totalWeight = totalweights[i];          // 初始化簇的总权重为节点的总权重
        c->nodes.insert(i);                        // 将节点添加到簇中
        RedBlackTree *rbTree = new RedBlackTree(); // 创建红黑树
        for (auto it = adjList[i]->begin(); it != adjList[i]->end(); it++)
        {
            double nodeiweight = totalweights[i];         // 节点i的总权重
            double nodejweight = totalweights[it->first]; // 节点j的总权重
            double wij = adjList[i]->at(it->first);       // 节点i到j的权重
            // 生成最大堆结点
            MHNode *mhnode = new MHNode();                           // 创建最大堆结点
            mhnode->i = i;                                           // 簇的id
            mhnode->j = it->first;                                   // 邻接表的key
            mhnode->delta = (2 * wij) / (nodeiweight + nodejweight); // 计算delta
            mh->insert(mhnode);                                      // 插入最大堆

            // 生成红黑树结点
            RBTNode *rbtnode = new RBTNode(); // 创建红黑树结点
            rbtnode->heap_ptr = mhnode;       // 将最大堆结点指针赋值给红黑树结点
            rbtnode->neighbor = it->first;    // 邻接表的key
            rbtnode->weight = it->second;     // 邻接表的value
            rbTree->insert(rbtnode);          // 插入红黑树
        }
        rbTrees.push_back(rbTree); // 将红黑树存储到数组中
        clusters.push_back(c);
    }
}

void HierarchicalClustering::performClustering()
{
    while (numClusters > numNodes * percent)
    {
        // 反复取最大堆顶，直到遇到有效的节点
        MHNode *maxNode = nullptr;
        while (true)
        {
            if (mh->heap.empty())
                break;
            maxNode = mh->getMax();
            // 检查两个簇是否都有效
            if (clusters[maxNode->i]->flag != 0 && clusters[maxNode->j]->flag != 0)
                break;
            // 否则移除该节点，继续
            mh->removeMax();
        }
        if (!maxNode || mh->heap.empty())
            break; // 堆空则退出

        int i = maxNode->i;
        int j = maxNode->j;
        if (i > j)
        {
            swap(i, j);
        }
        // cout << "i: " << i << " j: " << j << " delta: " << maxNode->delta << endl;
        mergeClusters(i, j);
    }
}

void HierarchicalClustering::mergeClusters(int i, int j)
{
    if (!rbTrees[i] || !rbTrees[j])
        return;

    RedBlackTree *rbTreei = rbTrees[i];
    RedBlackTree *rbTreej = rbTrees[j];

    // 1. 合并簇 j 的所有邻居到簇 i
    LinkListNode *jList = rbTreej->getLinkList();
    LinkListNode *curr = jList;
    double wij = 0; // 记录i-j边的权重

    while (curr != nullptr)
    {
        int k = curr->rbtnode->neighbor;
        double wjk = curr->rbtnode->weight;

        if (k == i)
        {
            // i 和 j 是邻居，这条边变成内部边
            wij = wjk;
            // 删除 i 红黑树中指向 j 的节点和最大堆节点
            RBTNode *ij = rbTreei->searchTree(j);
            if (ij)
            {
                if (ij->heap_ptr)
                    mh->deleteNode(ij->heap_ptr->index);
                rbTreei->deleteNode(j);
            }
            // 删除 j 红黑树中指向 i 的节点和最大堆节点（对称处理）
            RBTNode *ji = rbTreej->searchTree(i);
            if (ji)
            {
                if (ji->heap_ptr)
                    mh->deleteNode(ji->heap_ptr->index);
                rbTreej->deleteNode(i);
            }
            curr = curr->next;
            continue;
        }

        RBTNode *node_ik = rbTreei->searchTree(k);

        if (node_ik)
        {
            // i 和 j 都与 k 相邻，合并权重
            node_ik->weight += wjk;

            // 更新最大堆节点的 delta
            MHNode *mhNode = node_ik->heap_ptr;
            double wik = node_ik->weight;
            double wii = clusters[i]->inweight;
            double wkk = clusters[k]->inweight;
            double twi = clusters[i]->totalWeight;
            double twk = clusters[k]->totalWeight;
            double newDelta = (wii + wkk + 2 * wik) / (twi + twk) - wii / twi - wkk / twk;
            mhNode->delta = newDelta;
            mh->heapifyUp(mhNode->index);
            mh->heapifyDown(mhNode->index);
        }
        else
        {
            // k 只在 j，不在 i，插入新节点到 i 的红黑树和最大堆
            MHNode *mhNode = new MHNode();
            mhNode->i = i;
            mhNode->j = k;
            double nodeiweight = clusters[i]->totalWeight;
            double nodekweight = clusters[k]->totalWeight;
            mhNode->delta = (2 * wjk) / (nodeiweight + nodekweight);
            mh->insert(mhNode);

            RBTNode *rbtNode = new RBTNode();
            rbtNode->neighbor = k;
            rbtNode->weight = wjk;
            rbtNode->heap_ptr = mhNode;
            rbTreei->insert(rbtNode);
        }
        curr = curr->next;
    }

    // 释放链表内存
    curr = jList;
    while (curr)
    {
        LinkListNode *tmp = curr;
        curr = curr->next;
        delete tmp;
    }

    // 1.5 删除所有红黑树和最大堆中 neighbor 为 j 的节点（除了 i 和 j 自己）
    for (int idx = 0; idx < rbTrees.size(); ++idx)
    {
        if (rbTrees[idx] && idx != j && idx != i)
        {
            // 删除 idx->j
            RBTNode *node = rbTrees[idx]->searchTree(j);
            if (node)
            {
                if (node->heap_ptr)
                    mh->deleteNode(node->heap_ptr->index);
                rbTrees[idx]->deleteNode(j);
            }
            // 删除 j->idx（对称处理，防止堆中有 (j, idx) 组合）
            if (rbTrees[j])
            {
                RBTNode *node2 = rbTrees[j]->searchTree(idx);
                if (node2 && node2->heap_ptr)
                    mh->deleteNode(node2->heap_ptr->index);
            }
        }
    }

    // 2. 合并节点集合、权重
    clusters[i]->nodes.insert(clusters[j]->nodes.begin(), clusters[j]->nodes.end());
    clusters[i]->totalWeight += clusters[j]->totalWeight;

    // 3. 更新 inweight（i 与 j 之间的边权*2，加上 j 的内部权重）
    clusters[i]->inweight += clusters[j]->inweight + 2 * wij;

    // 4. 删除 i 到 j 的边（如果有），以及最大堆中的对应节点
    // 已在上面处理，无需重复

    // 5. 删除簇 j 的红黑树
    rbTreej->destroyTree(rbTreej->getRoot());
    rbTrees[j] = nullptr;

    // 6. 标记 j 已被合并
    clusters[j]->flag = 0;
    numClusters--;

    // 7. 更新所有最大堆中与 i 相关的节点的 delta，并删除所有与 j 相关的节点
    // 注意：遍历时不能直接删除元素，需先收集待删除的索引
    vector<int> toDelete;
    for (int idx = 0; idx < mh->heap.size(); ++idx)
    {
        MHNode *node = mh->heap[idx];
        if (node->i == j || node->j == j)
        {
            // 标记与j相关的节点，待会统一删除
            toDelete.push_back(idx);
            continue;
        }
        if (node->i == i || node->j == i)
        {
            int a = node->i, b = node->j;
            double wii = clusters[a]->inweight;
            double wjj = clusters[b]->inweight;
            double twi = clusters[a]->totalWeight;
            double twj = clusters[b]->totalWeight;
            RBTNode *rbtnode = rbTrees[a] ? rbTrees[a]->searchTree(b) : nullptr;
            double wij2 = rbtnode ? rbtnode->weight : 0;
            node->delta = (wii + wjj + 2 * wij2) / (twi + twj) - wii / twi - wjj / twj;
            mh->heapifyUp(node->index);
            mh->heapifyDown(node->index);
        }
    }
    // 倒序删除，避免索引混乱
    for (int k = toDelete.size() - 1; k >= 0; --k)
    {
        mh->deleteNode(toDelete[k]);
    }
}

void HierarchicalClustering::run(double &percent, string filename)
{
    this->percent = percent;
    readFile(filename);
    init();
    performClustering();
}