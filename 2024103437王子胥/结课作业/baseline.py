import elkai
import matplotlib.pyplot as plt
import numpy as np
from src.save_utils import save_detailed_result

class Baseline:
    """
    The template for all pipeline solving CMDTSP
    """
    def __init__(self, depots, cities, stations, graph, distance, full, limit, name):
        self.depots = depots
        self.cities = cities
        self.stations = stations
        self.graph = graph
        self.full = full
        self.limit = limit
        self.name = name
        self.distance = distance
        self.solution = []
        self.corner = [min([graph.nodes[i]['pos'][0] for i in graph.nodes]),
                       min([graph.nodes[i]['pos'][1] for i in graph.nodes])]
        self.run_time = 300

    def max_station(self):
        # return the max times a station is visited
        visited = [0 for _ in self.stations]
        for route in self.solution:
            for i in range(len(self.stations)):
                if self.stations[i] in route:
                    visited[i] += 1
        return max(visited) if len(visited) > 0 else 0

    def integer_location(self, location):
        # turn the coordinate from [-1, 1] to [0, +infinity)
        x, y = location[0] - self.corner[0], location[1] - self.corner[1]
        return [x * 10000, y * 10000]

    def lkh(self, group):
        # the lkh tsp solver, given a list of nodes, return the tsp circle in a list (end with start node)
        if len(group) == 1:
            return [group[0], group[0]]
        elif len(group) == 2:
            return [group[0], group[1], group[0]]
        else:
            visit = elkai.Coordinates2D(
                {i: self.integer_location(self.graph.nodes[i]['pos']) for i in group}).solve_tsp(runs=10)
            print(visit)  #这里是打印仓库访问路径，不含充电桩
            return visit

    def tour_length(self, tour):
        # given a tsp tour, return the length of the tour
        length = 0
        if len(tour) == 0:
            return length
        assert tour[0] == tour[-1], f"{tour} is not a tsp tour"
        for i, j in zip(tour[:-1], tour[1:]):
            length += self.distance[i][j]
        return length

    def solution_length(self, solution):
        # given a solution, return the sum of all tours
        length = 0
        for route in solution:
            length += self.tour_length(route)
        return length

    def output_path_costs(self):
        """
        输出每个仓库（子路径）的路径成本，并显示最长路径。
        """
        if not self.solution:
            print("未找到可行解。")
            return

        costs = []
        max_cost = -float('inf')
        max_cost_depot = None

        for idx, route in enumerate(self.solution):
            cost = self.tour_length(route)
            costs.append((f"仓库 {idx + 1}", cost))
            if cost > max_cost:
                max_cost = cost
                max_cost_depot = f"仓库 {idx + 1}"
#每个子路径成本
        #print("\n子路径成本统计：")
        #for depot, cost in costs:
        #   print(f"{depot} 的路径成本是: {cost:.2f}")
        print(f"最长路径为 {max_cost_depot}，路径成本是: {max_cost:.2f}")

    def is_feasible(self, solution):
        # check whether a solution is feasible (empty solution, empty tour, energy restriction, resource restriction)
        if len(solution) == 0:
            print('No solution')
            return False
        for tour in solution:
            # assert len(tour) > 1, f"should visit depot at least 2 times"
            if len(tour) < 2:
                print("An empty tour")
                return False
            energy = self.full
            for start, end in zip(tour[:-1], tour[1:]):
                energy -= self.distance[start][end]
                index = np.where(self.stations == end)[0].item() if end in self.stations else -1
                if index > -1:
                    energy = self.full
                if energy < 0:
                    print("Violate energy restriction")
                    return False
        if self.max_station() > self.limit:
            print("Visit one station too much times")
            return False
        return True

    def build_routes(self, paths):
        # construct a routes with stations from a tsp tour of cities
        routes = []
        for path in paths:
            route = [path[0]]
            energy = self.full
            next_customer = 1
            station_visited = [False for _ in self.stations]
            while route[-1] != path[-1] or len(route) < len(path):
                if energy > self.distance[route[-1]][path[next_customer]]:
                    energy -= self.distance[route[-1]][path[next_customer]]
                    route.append(path[next_customer])
                    next_customer += 1
                    station_visited = [False for _ in self.stations]
                else:
                    min_distance = 1e10
                    closest_station = None
                    for i in range(len(self.stations)):
                        if self.distance[self.stations[i]][route[-1]] < min_distance and not station_visited[i]:
                            min_distance = self.distance[self.stations[i]][route[-1]]
                            closest_station = i
                    if energy >= min_distance:
                        route.append(self.stations[closest_station])
                        energy = self.full
                        station_visited[closest_station] = True
                    else:
                        break
            if route[-1] == path[-1] and len(route) >= len(path):
                routes.append(route)
        return routes

    def show(self):
        # plot the solution
        plt.scatter([self.graph.nodes[x]['pos'][0] for x in self.depots],
                    [self.graph.nodes[x]['pos'][1] for x in self.depots], marker='o')
        plt.scatter([self.graph.nodes[x]['pos'][0] for x in self.cities],
                    [self.graph.nodes[x]['pos'][1] for x in self.cities], marker='x')
        plt.scatter([self.graph.nodes[x]['pos'][0] for x in self.stations],
                    [self.graph.nodes[x]['pos'][1] for x in self.stations], marker='*')
        for route in self.solution:
            plt.plot([self.graph.nodes[x]['pos'][0] for x in route], [self.graph.nodes[x]['pos'][1] for x in route])
            energy = self.full
            for i in range(len(route) - 1):
                point = self.graph.nodes[route[i]]['pos'][0]
                plt.text(point[0], point[1], f"{energy: .2f}")
                if i < len(route) - 1:
                    if route[i] in self.stations:
                        energy = self.full
                    energy -= self.distance[route[i]][route[i + 1]]
        plt.title(self.name)
        plt.tight_layout()
        plt.show()
