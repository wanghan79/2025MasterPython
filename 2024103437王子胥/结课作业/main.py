import networkx as nx
import numpy as np
import pickle
from src.ACO import ACO
from src.EA import EA
from src.MILP import MILP
from src.VNS import VNS
from src.PTC import PTC
from src.data import manhattan, boston, md_vrp, prepare_data
import time
from tqdm import tqdm


def evaluate_baseline(name, d, c, s, g, distance, full, limit, method='lkh', part='mtsp'):
    if name == 'Hierarchical Pipeline':
        baseline = PTC(d, c, s, g, distance, full, limit, method=method)
    elif name == 'Variable Neighborhood Search':
        baseline = VNS(d, c, s, g, distance, full, limit)
    elif name == 'Evolutionary Algorithm':
        baseline = EA(d, c, s, g, distance, full, limit)
    elif name == 'Ant Colony Optimization':
        baseline = ACO(d, c, s, g, distance, full, limit)
    elif name == 'Mixture Integer Linear Programming':
        baseline = MILP(d, c, s, g, distance, full, limit)
    else:
        assert False, f"Baseline {name} Not Implemented"
    start = time.time()
    baseline.solve()
    end = time.time()
    print(f'baseline {baseline.name} use {end - start:.2f}s to get '
          f'{"feasible" if baseline.is_feasible(baseline.solution) else "infeasible"} solution with tour length '
          f'{baseline.solution_length(baseline.solution):.2f}, visit station at most {baseline.max_station()} times')
    print('-------------------------------------------------------------------------')
    return end - start, baseline.solution_length(baseline.solution), baseline.is_feasible(baseline.solution)


def run_baselines(num=1000):
    for filename in [#'manhattan',
         #'cambridge',
         #'mdvrp',
        #'random',
                     'manhattan_large',
        #'cambridge_large',
         #'mdvrp_large',
        # 'random_large'
    ]:
        with open(f'datasets/{filename}.pickle', 'rb') as f:
            dataset = pickle.load(f)
        stats = {'Hierarchical Pipeline': {"Length": [], "Feasible": [], "Time": []},
                 #'Variable Neighborhood Search': {"Length": [], "Feasible": [], "Time": []},
                 #'Evolutionary Algorithm': {"Length": [], "Feasible": [], "Time": []},
                 #'Ant Colony Optimization': {"Length": [], "Feasible": [], "Time": []}
                 }
        graph = manhattan() if 'manhattan' in filename else boston()[0] if 'cambridge' in filename else md_vrp()[0] \
            if 'mdvrp' in filename else nx.Graph()
        full = 4 if 'manhattan' in filename or 'mdvrp' in filename else 40 if 'cambridge' in filename else 1
        limit = 2
        for data in tqdm(dataset[:min(num, len(dataset))]):
            d, c, s, distance = data
            depots, cities, stations = np.array(d), np.array(c), np.array(s)
            if 'random' in filename:
                locations = np.concatenate((depots, cities, stations))
                for i in range(len(locations)):
                    graph.add_node(i, pos=locations[i].tolist())
                for i in range(len(locations)):
                    for j in range(i, len(locations)):
                        graph.add_edge(i, j, weight=distance[i][j])
                depots = np.array([i for i in range(len(depots))])
                cities = np.array([i + len(depots) for i in range(len(cities))])
                stations = np.array([i + len(depots) + len(cities) for i in range(len(stations))])
            if 'mdvrp' in filename:
                g = nx.Graph()
                locations = np.concatenate((depots, cities, stations))
                for i in locations:
                    g.add_node(i, pos=graph.nodes[i]['pos'])
                for i in locations:
                    for j in locations:
                        g.add_edge(i, j, weight=graph.edges[i, j]['weight'])
            for name in stats.keys():
                running_time, tour_length, feasible = \
                    evaluate_baseline(name, depots, cities, stations, graph, distance, full, limit)
                if feasible:
                    stats[name]['Length'].append(tour_length)
                    stats[name]['Feasible'].append(1)
                    stats[name]['Time'].append(running_time)
        with open(f'datasets/{filename}_stat.pickle', 'wb') as f:
            pickle.dump(stats, f)

        def avg(x):
            return sum(x) / len(x) if len(x) > 0 else 0

        for name in stats.keys():
            print(f"baseline {name} use {avg(stats[name]['Time']):.2f}s to get {avg(stats[name]['Length']):.2f} tour "
                  f"with feasibility rate {sum(stats[name]['Feasible']) / num:.2f} on dataset {filename}")


def run_ablation_tsp():
    with open('datasets/random.pickle', 'rb') as f:
        dataset = pickle.load(f)
    graph, full, limit = nx.Graph(), 1, 2
    stats = {'lkh': {"Length": [], "Time": []}, 'exact': {"Length": [], "Time": []}, 'appr': {"Length": [], "Time": []},
             'meta': {"Length": [], "Time": []}, 'neural': {"Length": [], "Time": []}}
    for data in tqdm(dataset):
        d, c, s, distance = data
        depots, cities, stations = np.array(d), np.array(c), np.array(s)
        locations = np.concatenate((depots, cities, stations))
        for i in range(len(locations)):
            graph.add_node(i, pos=locations[i].tolist())
        for i in range(len(locations)):
            for j in range(i, len(locations)):
                graph.add_edge(i, j, weight=distance[i][j])
        depots = np.array([i for i in range(len(depots))])
        cities = np.array([i + len(depots) for i in range(len(cities))])
        stations = np.array([i + len(depots) + len(cities) for i in range(len(stations))])
        for method in ['lkh', 'exact', 'appr', 'meta']:
            name = 'Hierarchical Pipeline'
            running_time, tour_length, _ = \
                evaluate_baseline(name, depots, cities, stations, graph, distance, full, limit, method)
            stats[method]['Time'].append(running_time)
            stats[method]['Length'].append(tour_length)

    with open(f'datasets/ablation_tsp_stat.pickle', 'wb') as f:
        pickle.dump(stats, f)


def run_ablation_partition():
    with open('datasets/random.pickle', 'rb') as f:
        dataset = pickle.load(f)
    graph, full, limit = nx.Graph(), 1, 2
    stats = {'mtsp': {"Length": []}, 'nn': {"Length": []}, 'kmean': {"Length": []}}
    for data in tqdm(dataset):
        d, c, s, distance = data
        depots, cities, stations = np.array(d), np.array(c), np.array(s)
        locations = np.concatenate((depots, cities, stations))
        for i in range(len(locations)):
            graph.add_node(i, pos=locations[i].tolist())
        for i in range(len(locations)):
            for j in range(i, len(locations)):
                graph.add_edge(i, j, weight=distance[i][j])
        depots = np.array([i for i in range(len(depots))])
        cities = np.array([i + len(depots) for i in range(len(cities))])
        stations = np.array([i + len(depots) + len(cities) for i in range(len(stations))])
        for part in ['mtsp', 'nn', 'kmean']:
            name = 'Hierarchical Pipeline'
            _, tour_length, _ = \
                evaluate_baseline(name, depots, cities, stations, graph, distance, full, limit, method='lkh', part=part)
            stats[part]['Length'].append(tour_length)

    with open(f'datasets/ablation_partition_stat.pickle', 'wb') as f:
        pickle.dump(stats, f)


def run_ablations():
    run_ablation_tsp()
    run_ablation_partition()


if __name__ == '__main__':
    prepare_data(1000)
    run_baselines(1000)
    run_ablations()
