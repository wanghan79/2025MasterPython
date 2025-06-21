import sys
import time
from python.hcluster import HierarchicalClustering
from python.vns import VNS, Parameters


def run_clustering(k, filename, percent, limitTime):
    """运行聚类算法"""
    print(f"参数设置:")
    print(f"  目标簇数量: {k}")
    print(f"  输入文件: {filename}")
    print(f"  压缩比例: {percent}")
    print(f"  时间限制: {limitTime} 秒")
    print("-" * 50)
    
    # 第一阶段：层次聚类
    print("第一阶段：执行层次聚类...")
    start_time = time.time()
    
    hc = HierarchicalClustering(percent)
    hc.readFile(filename)
    print(f"读取图文件成功: {hc.numNodes} 个节点, {hc.numEdges} 条边")
    
    hc.init()
    hc.performClustering()
    
    hc_time = time.time() - start_time
    print(f"层次聚类完成，耗时: {hc_time:.2f} 秒")
    
    # 统计有效簇数量
    valid_clusters = []
    for i, cluster in enumerate(hc.clusters):
        if cluster.flag == 1:
            valid_clusters.append((i, cluster))
    
    print(f"层次聚类结果: {len(valid_clusters)} 个有效簇")
    
    # 如果层次聚类的簇数量已经小于等于目标k，直接输出结果
    if len(valid_clusters) <= k:
        print(f"层次聚类结果已满足要求（{len(valid_clusters)} <= {k}），无需VNS优化")
        print_clustering_results(valid_clusters)
        return
    
    # 第二阶段：VNS优化
    print(f"\n第二阶段：VNS算法优化（目标簇数: {k}）...")
    
    # 准备VNS参数
    params = Parameters()
    params.k = k
    params.numNodes = hc.numNodes
    params.numEdges = hc.numEdges
    params.tMin = min(20, hc.numNodes // 100 + 1)
    params.tStep = params.tMin
    params.tMax = min(200, hc.numNodes // 5 + 1)
    params.limitTime = limitTime
    params.filename = filename
    params.adjList = hc.adjList
    params.nodeWeight = hc.totalweights.copy()
    
    # 设置层次聚类结果
    params.hcResults = {}
    for i, cluster in valid_clusters:
        params.hcResults[i] = cluster
    
    # 设置红黑树结果
    params.rbTreesResult = {}
    for i, tree in enumerate(hc.rbTrees):
        if tree is not None:
            params.rbTreesResult[i] = tree
    
    print(f"VNS参数设置: tMin={params.tMin}, tMax={params.tMax}, tStep={params.tStep}")
    
    # 运行VNS算法
    vns_start_time = time.time()
    vns = VNS(params)
    vns.run_vns()
    vns_time = time.time() - vns_start_time
    
    print(f"VNS算法完成，耗时: {vns_time:.2f} 秒")
    
    # 输出最终结果
    print(f"\n最终聚类结果:")
    final_clusters = []
    for cluster_id, cluster in vns.clusters.items():
        final_clusters.append((cluster_id, cluster))
    
    print_clustering_results(final_clusters)
    
    # 计算并输出质量指标
    final_ncut = vns.calculate_ncut()
    print(f"\n质量指标:")
    print(f"  最终NCut值: {final_ncut:.6f}")
    print(f"  簇数量: {len(final_clusters)}")
    
    # 输出总时间统计
    total_time = time.time() - start_time
    print(f"\n时间统计:")
    print(f"  层次聚类时间: {hc_time:.2f} 秒")
    print(f"  VNS优化时间: {vns_time:.2f} 秒")
    print(f"  总耗时: {total_time:.2f} 秒")


def print_clustering_results(clusters):
    """打印聚类结果"""
    print(f"共 {len(clusters)} 个簇:")
    
    for i, (cluster_id, cluster) in enumerate(sorted(clusters)):
        if hasattr(cluster, 'nodes'):
            nodes = sorted(list(cluster.nodes))
        else:
            nodes = sorted(list(cluster))
        
        print(f"  簇 {i+1} (ID:{cluster_id}): {nodes} ({len(nodes)} 个节点)")
        
        # 如果有权重信息，也打印出来
        if hasattr(cluster, 'totalWeight'):
            print(f"    总权重: {cluster.totalWeight:.3f}")
        if hasattr(cluster, 'inweight'):
            print(f"    内部权重: {cluster.inweight:.3f}")


if __name__ == "__main__":
    # 默认参数设置
    default_k = 3
    default_filename = "instances/karate.graph"
    default_percent = 0.5
    default_limitTime = 30
    
    # 检查命令行参数
    if len(sys.argv) == 1:
        # 无参数，使用默认值
        print("未提供参数，使用默认测试参数...")
        k = default_k
        filename = default_filename
        percent = default_percent
        limitTime = default_limitTime
        
    elif len(sys.argv) == 5:
        # 完整参数
        k = int(sys.argv[1])
        filename = sys.argv[2]
        percent = float(sys.argv[3])
        limitTime = int(sys.argv[4])
        
    else:
        # 参数数量不对，显示使用说明
        print("使用方法:")
        print(f"  python {sys.argv[0]} <k> <filename> <percent> <limitTime>")
        print(f"  或者直接运行: python {sys.argv[0]} (使用默认参数)")
        print("")
        print("参数说明:")
        print("  k: 目标簇数量")
        print("  filename: 图文件路径")
        print("  percent: 层次聚类压缩比例 (0.0-1.0)")
        print("  limitTime: VNS算法时间限制(秒)")
        print("")
        print("默认参数:")
        print(f"  k = {default_k}")
        print(f"  filename = {default_filename}")
        print(f"  percent = {default_percent}")
        print(f"  limitTime = {default_limitTime}")
        print("")
        print("使用示例:")
        print(f"  python {sys.argv[0]} 3 instances/karate.graph 0.5 30")
        print(f"  python {sys.argv[0]} 5 instances/dolphins.graph 0.3 60")
        sys.exit(1)
    
    # 运行聚类算法
    run_clustering(k, filename, percent, limitTime)