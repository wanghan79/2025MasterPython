#include "vns.h"
#include "hierarchicalClustering.h"
#include "maxHeap.h"
#include "redBlackTree.h"

int main(int argc, char *argv[])
{
    // 主函数传参数，k，filename
    struct parameters params;
    string filename = "";
    double percent = 0.5;
    if (argc >= 5)
    {
        params.k = atoi(argv[1]);
        filename = argv[2];
        percent = atof(argv[3]);
        params.limitTime = atoi(argv[4]);
        params.filename = filename;
    }
    else
    {
        std::cerr << "Usage: " << argv[0] << " <k> <filename> <percent> <limitTime> " << std::endl;
        return 1;
    }

    HierarchicalClustering hc;
    hc.run(percent, filename);

    params.numNodes = hc.numNodes;
    params.numEdges = hc.numEdges;

    params.tMin = std::min(20, params.numNodes / 100 + 1);
    params.tStep = params.tMin;
    params.tMax = std::min(200, static_cast<int>(params.numNodes) / 5 + 1);
    params.adjList = hc.adjList;
    params.nodeWeight = hc.totalweights;

    int hcsize = hc.clusters.size();
    for (int i = 0; i < hcsize; i++)
    {
        if (hc.clusters[i]->flag == 0)
            continue; // 只输出有效的簇
        params.hcResults.insert({i, hc.clusters[i]});
    }

    int rbsize = hc.rbTrees.size();

    for (int i = 0; i < rbsize; i++)
    {
        if (hc.rbTrees[i] == nullptr)
            continue; // 只输出有效的红黑树
        params.rbTreesResult.insert({i, hc.rbTrees[i]});
    }

    vns v(params);
    v.runVNS();
    return 0;
}
