#include "vns.h"
#include <iomanip>

vns::vns(struct parameters &params)
    : params(params)
{
    clusters.clear();
}
void vns::initParamers()
{
    k = params.k;
    numNodes = params.numNodes;
    numEdges = params.numEdges;
    tMin = params.tMin;
    tMax = params.tMax;
    tStep = params.tStep;
    nodeWeight = params.nodeWeight; // 初始化节点权重
    limitTime = params.limitTime;   // 初始化限制时间
    // 初始化簇并移除 nullptr 簇
    for (auto &pair : params.hcResults)
    {
        if (pair.second != nullptr && pair.second->flag == 1)
        {
            clusters.insert(pair); // 仅插入非 nullptr 的簇
        }
    }

    // 初始化邻接表
    adjList = params.adjList; // 邻接表
    // 初始化红黑树 遍历簇并处理对应的红黑树
    for (auto &pair : params.rbTreesResult)
    {
        if (clusters.find(pair.first) != clusters.end())
        {
            rbTrees.insert(pair); // 仅插入非 nullptr 的红黑树
        }
    }
}
void vns::initResults() // 初始化解
{
    int size = clusters.size();
    while (size > k)
    {
        int clusteri = 0;
        int clusterj = 0;
        findMerge(clusteri, clusterj);     // 找到最好的合并,当前是随机合并
        mergeClusters(clusteri, clusterj); // 合并簇
        size = clusters.size();            // 更新簇的数量
    }
    calculateSAllWeight(); // 计算所有簇之间权重
    nodesID.resize(numNodes);
    for (int i = 0; i < numNodes; i++)
    {
        nodesID[i] = i;
    }
}

void vns::runVNS()
{
    initParamers();
    initResults();

    int t = tMin;
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

    double currentNcut = calculateNCut(); // 当前ncut
    double lastNcut = 999;                // 上一次ncut
    double improveNCutChange = 999;       // 改进的ncut变化
    srand(time(0));
    while (duration < limitTime)
    {

        if (rand() % 100 > 99) // 95%概率进行快速局部搜索，5%概率进行完整局部搜索
        {
            cout << " CL ";
            // shaking(t); // 扰动
            completedLocalSearch();
        }
        else
        {
            cout << " FL ";
            fastLocalSearch();
        }
        lastNcut = currentNcut; // 保存上一次ncut
        currentNcut = calculateNCut();
        improveNCutChange = lastNcut - currentNcut; // 计算改进的ncut变化
        if (tempNCutChange <= bestNCutChange && improveNCutChange >= 0.0001)
        {
            bestNCutChange = tempNCutChange;
            t = tMin;
        }
        else
        {
            t += tStep;
        }
        if (t >= tMax)
        {
            t = tMin;
        }
        shaking(t);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
        cout << fixed << setprecision(6)
             << "ncut: " << currentNcut
             << setw(9) << "dleta: " << improveNCutChange << " "
             << "t: " << t << setw(7)
             << "time: " << duration << endl;
    }
}

void vns::mergeClusters(int i, int j)
{
    // 更新被合并簇j中所有节点的label为i
    for (auto node : clusters[j]->nodes)
    {
        label[node] = i;
    }
    clusters[i]->nodes.insert(clusters[j]->nodes.begin(), clusters[j]->nodes.end());
    clusters.erase(j);
}
void vns::shaking(int t)
{
    srand(time(0)); // 随机数种子
    // 随机选择两个簇进行交换两个节点t次
    vector<int> keys;
    keys.resize(clusters.size());
    int i = 0;
    for (auto it : clusters)
    {
        keys[i] = it.first;
        i++;
    }
    while (t-- > 0)
    { // 从0到clusters.size()-1，随机生成两个整数，包括0和clusters.size()-1
        int idx1 = rand() % clusters.size();
        int idx2 = rand() % clusters.size();
        while (idx1 == idx2)
        {
            idx2 = rand() % clusters.size();
        }
        // 从id1簇里面随机选择一个点
        auto it = clusters[keys[idx1]]->nodes.begin();
        std::advance(it, rand() % clusters[keys[idx1]]->nodes.size());
        int nodei = *it;

        // 从id2簇里面随机选择一个点
        auto it2 = clusters[keys[idx2]]->nodes.begin();
        std::advance(it2, rand() % clusters[keys[idx2]]->nodes.size());
        int nodej = *it2;
        // 交换节点 i 和 j
        swapNodes(nodei, nodej); // 交换节点 i 和 j
    }
}

void vns::fastLocalSearch()
{
    bool flag = true;
    while (flag)
    {
        flag = false;
        int nodei = -1;
        int clusterj = -1;

        // 随机打乱id，从随机的id开始遍历，查找到第一个可以移动的节点
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(nodesID.begin(), nodesID.end(), g);

        for (auto n : nodesID)
        {
            int forFlag = 0;
            for (auto c : clusters)
            {
                if (label[n] == c.first)
                    continue;
                bool isNeighbor = isneighbor(n, c.first);
                if (!isNeighbor)
                    continue;
                double currentChange = calculateNCutChange(n, c.first);
                if (isNeighbor && currentChange < 0)
                {
                    tempNCutChange = currentChange;
                    nodei = n;
                    clusterj = c.first;
                    forFlag = 1;
                    break;
                }
            }
            if (forFlag)
                break;
        }
        if (nodei == -1 && clusterj == -1)
            break;
        else
        {
            moveNode(nodei, clusterj);
            flag = true;
        }
    }
}

void vns::completedLocalSearch()
{
    bool flag = true;
    while (flag)
    {
        flag = false;
        int nodei = -1;
        int clusterj = -1;
        double bestChange = 0.0; // record this iteration best change
        // 随机打乱id，从随机的id开始遍历，查找到第一个可以移动的节点
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(nodesID.begin(), nodesID.end(), g);

        for (auto n : nodesID)
        {
            for (auto c : clusters)
            {
                if (label[n] == c.first)
                    continue;
                double currentChange = calculateNCutChange(n, c.first);
                if (currentChange < bestChange && currentChange < 0)
                {
                    bestChange = currentChange;
                    tempNCutChange = currentChange;
                    nodei = n;
                    clusterj = c.first;
                }
            }
        }
        if (nodei == -1 && clusterj == -1)
            break;
        else
        {
            moveNode(nodei, clusterj);
            flag = true;
        }
    }
}

double vns::calculateNCutChange(int node, int cluster)
{
    double cutChange = 0.0;
    double oldCut = 0.0;
    double newCut = 0.0;

    double ci = label[node];                // 节点所属的簇
    double cj = cluster;                    // 目标簇
    double owi = clusters[ci]->outweight;   // 簇 i 的外部权重
    double owj = clusters[cj]->outweight;   // 簇 j 的外部权重
    double iwi = clusters[ci]->inweight;    // 簇 i 的内部权重
    double iwj = clusters[cj]->inweight;    // 簇 j 的内部权重
    double twi = clusters[ci]->totalWeight; // 簇 i 的总权重
    double twj = clusters[cj]->totalWeight; // 簇 j 的总权重
    // 计算旧的cut值
    oldCut = owi / twi + owj / twj;

    // 计算i移除节点的内部权重
    for (auto n : clusters[ci]->nodes)
    {
        if ((*adjList[node]).find(n) != (*adjList[node]).end())
        {
            double w = (*adjList[node])[n];
            iwi -= 2 * w;
        }
    }
    // 计算i移除节点的总权重
    twi -= nodeWeight[node];
    // 计算i移除节点的外部权重
    owi = twi - iwi;

    // 计算j添加节点的内部权重
    for (auto n : clusters[cj]->nodes)
    {
        if ((*adjList[node]).find(n) != (*adjList[node]).end())
        {
            double w = (*adjList[node])[n];
            iwj += 2 * w;
        }
    }
    // 计算j添加节点的总权重
    twj += nodeWeight[node];
    // 计算j添加节点的外部权重
    owj = twj - iwj;
    // 计算新的cut值
    newCut = owi / twi + owj / twj;
    // 计算cut值变化
    cutChange = newCut - oldCut;
    return cutChange;
}

void vns::findMerge(int &clusteri, int &clusterj)
{
    // 随机数种子
    srand(time(0));
    std::vector<int> keys;
    for (const auto &kv : clusters)
    {
        if (kv.second != nullptr)
            keys.push_back(kv.first);
    }
    if (keys.size() < 2)
        return;
    int idx1 = rand() % keys.size();
    int idx2 = rand() % (keys.size() - 1);
    if (idx2 >= idx1)
        idx2++; // 保证idx1和idx2不同
    clusteri = keys[idx1];
    clusterj = keys[idx2];
}

void vns::calculateSAllWeight()
{
    // 计算节点权重
    for (int nodeID = 0; nodeID < adjList.size(); nodeID++)
    {
        double tw = 0.0;
        for (auto node : *adjList[nodeID])
        {
            if (node.first != nodeID)
            {
                tw += node.second;
            }
        }
        nodeWeight[nodeID] = tw; // 节点权重
    }
    // 计算总权重
    for (auto it : clusters)
    {
        double totalWeight = 0.0;
        for (auto node : it.second->nodes)
        {
            totalWeight += nodeWeight[node];
            label[node] = it.first;
        }
        it.second->totalWeight = totalWeight;
    }

    // 计算内部权重和外部权重
    for (auto it : clusters)
    {
        double inWeight = 0.0;
        for (auto nodei : it.second->nodes)
        {
            for (auto nodej : it.second->nodes)
            {
                if (nodei == nodej || nodei > nodej) // 避免重复计算
                    continue;
                if ((*adjList[nodei]).find(nodej) != (*adjList[nodei]).end())
                {
                    double w = (*adjList[nodei])[nodej];
                    inWeight += w;
                }
            }
        }
        it.second->inweight = 2 * inWeight; // 每条边都被计算了一次
        it.second->outweight = it.second->totalWeight - inWeight;
    }
}

double vns::calculateNCut()
{
    // 计算并输出ncut值
    double ncut = 0.0;
    for (auto it : clusters)
    {
        ncut += it.second->outweight / it.second->totalWeight;
    }
    // cout << "ncut: " << ncut;
    return ncut;
}

void vns::moveNode(int node, int cluster)
{
    int nodei = node;
    int clusteri = label[nodei];
    int clusterj = cluster;

    double inWeightNodei = 0.0; // 节点 i 在簇 i 中的内部权重
    double inWeightNodej = 0.0; // 节点 i 在簇 j 中的内部权重

    // 计算节点 i 在簇 i 中的权重
    for (auto node : clusters[clusteri]->nodes)
    {
        if ((*adjList[nodei]).find(node) != (*adjList[nodei]).end())
        {
            double w = (*adjList[nodei])[node];
            inWeightNodei += w;
        }
    }
    // clusteri 的内部权重减去节点 i 的内部权重，要减去 2 倍
    clusters[clusteri]->inweight -= inWeightNodei * 2;
    // 计算节点 i 在簇 j 中的权重
    for (auto node : clusters[clusterj]->nodes)
    {
        if ((*adjList[nodei]).find(node) != (*adjList[nodei]).end())
        {
            double w = (*adjList[nodei])[node];
            inWeightNodej += w;
        }
    }
    // clusterj 的内部权重加上节点 i 的内部权重，要加上 2 倍
    clusters[clusterj]->inweight += inWeightNodej * 2;
    // 计算总权重
    clusters[clusteri]->totalWeight -= nodeWeight[nodei];
    clusters[clusterj]->totalWeight += nodeWeight[nodei];
    // 更新簇的外部权重
    clusters[clusteri]->outweight = clusters[clusteri]->totalWeight - clusters[clusteri]->inweight;
    clusters[clusterj]->outweight = clusters[clusterj]->totalWeight - clusters[clusterj]->inweight;

    // 更新簇的节点集合
    clusters[clusteri]->nodes.erase(nodei);  // 删除节点 i
    clusters[clusterj]->nodes.insert(nodei); // 添加节点 i
    label[nodei] = clusterj;                 // 更新标签
}

void vns::swapNodes(int i, int j)
{
    int nodei = i;
    int nodej = j;
    int clusteri = label[nodei];
    int clusterj = label[nodej];

    // 维护节点集合
    clusters[clusteri]->nodes.erase(nodei);
    clusters[clusterj]->nodes.erase(nodej);

    // 计算节点 i在cluster i 中的权重
    double inWeightNodei_inI = 0.0;
    for (auto node : clusters[clusteri]->nodes)
        if ((*adjList[nodei]).find(node) != (*adjList[nodei]).end())
            inWeightNodei_inI += (*adjList[nodei])[node];
    clusters[clusteri]->inweight -= inWeightNodei_inI * 2;

    // 计算节点 j在cluster j 中的权重
    double inWeightNodej_inJ = 0.0;
    for (auto node : clusters[clusterj]->nodes)
        if ((*adjList[nodej]).find(node) != (*adjList[nodej]).end())
            inWeightNodej_inJ += (*adjList[nodej])[node];
    clusters[clusterj]->inweight -= inWeightNodej_inJ * 2;

    // 计算节点 i在cluster j 中的权重
    double inWeightNodei_inJ = 0.0;
    for (auto node : clusters[clusterj]->nodes)
        if ((*adjList[nodei]).find(node) != (*adjList[nodei]).end())
            inWeightNodei_inJ += (*adjList[nodei])[node];
    clusters[clusterj]->inweight += inWeightNodei_inJ * 2;

    // 计算节点 j在cluster i 中的权重
    double inWeightNodej_inI = 0.0;
    for (auto node : clusters[clusteri]->nodes)
        if ((*adjList[nodej]).find(node) != (*adjList[nodej]).end())
            inWeightNodej_inI += (*adjList[nodej])[node];
    clusters[clusteri]->inweight += inWeightNodej_inI * 2;

    // 计算总权重
    clusters[clusteri]->totalWeight -= nodeWeight[nodei];
    clusters[clusteri]->totalWeight += nodeWeight[nodej];
    clusters[clusterj]->totalWeight -= nodeWeight[nodej];
    clusters[clusterj]->totalWeight += nodeWeight[nodei];

    // 更新簇的外部权重
    clusters[clusteri]->outweight = clusters[clusteri]->totalWeight - clusters[clusteri]->inweight;
    clusters[clusterj]->outweight = clusters[clusterj]->totalWeight - clusters[clusterj]->inweight;

    // 更新label和簇的点
    clusters[clusteri]->nodes.insert(nodej);
    clusters[clusterj]->nodes.insert(nodei);
    label[nodei] = clusterj;
    label[nodej] = clusteri;
}

bool vns::isneighbor(int nodei, int clusterj)
{
    // 判断nodei是否和clusterj相邻
    for (auto node : clusters[clusterj]->nodes)
    {
        if ((*adjList[nodei]).find(node) != (*adjList[nodei]).end())
        {
            return true;
        }
    }
    return false;
}