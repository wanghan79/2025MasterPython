# baseline of evolutionary algorithm from https://doi.org/10.3390/su12052127

from src.baseline import Baseline
import copy
import numpy as np
import time


class EA(Baseline):
    def __init__(self, depots, cities, stations, graph, distance, full, limit):
        super().__init__(depots, cities, stations, graph, distance, full, limit, 'Evolutionary Algorithm')
        self.size = 5  # size of population
        self.theta = 2  # max iters without improvement
        self.iters = 20
        self.omega = 3  # shaking strength
        self.lam = (0.1, 0.2)
        self.alpha = 5
        self.beta = 0.1
        self.population = []

    def initial_population(self):
        solution = [[depot] for depot in self.depots]
        indices = np.random.choice([i for i in range(len(self.cities))], len(self.depots), replace=False)
        for i in range(len(self.depots)):
            solution[i].append(self.cities[indices[i]])
        un_visited_cities = [i for i in range(len(self.cities)) if i not in indices]
        while len(un_visited_cities) > 0:
            closest_city, closest_distance, closest_depot = None, 100000000, None
            for i in un_visited_cities:
                city = self.cities[i]
                for j in range(len(self.depots)):
                    end = solution[j][-1]
                    if self.distance[city][end] < closest_distance:
                        closest_city = i
                        closest_distance = self.distance[city][end]
                        closest_depot = j
            solution[closest_depot].append(self.cities[closest_city])
            un_visited_cities.remove(closest_city)
        for i in range(len(self.depots)):
            solution[i].append(solution[i][0])
        self.population.append(solution)

    def variable_neighborhood_search(self, solution):
        cost = self.solution_cost(solution)
        optimal_solution, basic_solution = copy.deepcopy(solution), copy.deepcopy(solution)
        k = 0
        while k < 4:
            best_neighbor = self.local_search(basic_solution, k)
            if self.solution_cost(best_neighbor) < cost:
                optimal_solution = copy.deepcopy(best_neighbor)
                cost = self.solution_cost(optimal_solution)
                k = -1
            k += 1
            basic_solution = self.shake(basic_solution, optimal_solution)
        return optimal_solution

    def local_search(self, solution, k):
        best_neighbor = copy.deepcopy(solution)
        if k == 0:
            optimal, value = None, self.solution_cost(solution)
            for i in range(len(self.depots)):
                for j in range(1, len(best_neighbor[i]) - 1):
                    city = best_neighbor[i].pop(j)
                    for m in range(1, len(best_neighbor[i])):
                        best_neighbor[i].insert(m, city)
                        if self.solution_cost(best_neighbor) < value:
                            value = self.solution_cost(best_neighbor)
                            optimal = copy.deepcopy(best_neighbor)
                        best_neighbor[i].pop(m)
                    best_neighbor[i].insert(j, city)
            best_neighbor = optimal if optimal is not None else best_neighbor
        elif k == 1:
            optimal, value = None, self.solution_cost(solution)
            for i in range(len(self.depots)):
                for j in range(1, len(best_neighbor[i]) - 1):
                    for m in range(j + 1, len(best_neighbor[i]) - 1):
                        tmp = copy.deepcopy(best_neighbor[i][j])
                        best_neighbor[i][j] = copy.deepcopy(best_neighbor[i][m])
                        best_neighbor[i][m] = tmp
                        if self.solution_cost(best_neighbor) < value:
                            value = self.solution_cost(best_neighbor)
                            optimal = copy.deepcopy(best_neighbor)
                        tmp = copy.deepcopy(best_neighbor[i][j])
                        best_neighbor[i][j] = copy.deepcopy(best_neighbor[i][m])
                        best_neighbor[i][m] = tmp
            best_neighbor = optimal if optimal is not None else best_neighbor
        elif k == 2:
            optimal, value = None, self.solution_cost(solution)
            for i in range(len(self.depots)):
                for j in range(1, len(best_neighbor[i]) - 1):
                    city = best_neighbor[i].pop(j)
                    for m in range(len(self.depots)):
                        if m == i:
                            continue
                        for n in range(1, len(best_neighbor[m])):
                            best_neighbor[m].insert(n, city)
                            if self.solution_cost(best_neighbor) < value:
                                value = self.solution_cost(best_neighbor)
                                optimal = copy.deepcopy(best_neighbor)
                            best_neighbor[m].pop(n)
                    best_neighbor[i].insert(j, city)
            best_neighbor = optimal if optimal is not None else best_neighbor
        elif k == 3:
            optimal, value = None, self.solution_cost(solution)
            for i in range(len(self.depots)):
                for j in range(i + 1, len(self.depots)):
                    for m in range(1, len(best_neighbor[i]) - 1):
                        for n in range(1, len(best_neighbor[j]) - 1):
                            tmp = copy.deepcopy(best_neighbor[i][m])
                            best_neighbor[i][m] = copy.deepcopy(best_neighbor[j][n])
                            best_neighbor[j][n] = tmp
                            if self.solution_cost(best_neighbor) < value:
                                value = self.solution_cost(best_neighbor)
                                optimal = copy.deepcopy(best_neighbor)
                            tmp = copy.deepcopy(best_neighbor[i][m])
                            best_neighbor[i][m] = copy.deepcopy(best_neighbor[j][n])
                            best_neighbor[j][n] = tmp
            best_neighbor = optimal if optimal is not None else best_neighbor
        return best_neighbor

    def shake(self, current_solution, optimal_solution):
        basic_solution = copy.deepcopy(current_solution)
        omega = 0
        while omega < self.omega:
            omega += 1
            lam = np.random.random() * (self.lam[1] - self.lam[0]) + self.lam[0]
            neighbor = np.random.choice([5, 6], 1).item()
            solution = copy.deepcopy(basic_solution)
            if neighbor == 5 or len([tour for tour in solution if len(tour) > 3]) < 2:
                index = np.random.choice([i for i in range(len(self.depots)) if len(solution[i]) > 3], 1).item()
                tour = solution[index]
                interval = np.random.choice([i for i in range(1, len(tour) - 1)], 2, replace=False)
                i, j = min(interval[0], interval[1]), max(interval[0], interval[1])
                tour[i: j] = reversed(tour[i: j])
            else:
                indices = np.random.choice([i for i in range(len(self.depots)) if len(solution[i]) > 3], 2,
                                           replace=False)
                tour_1, tour_2 = solution[indices[0]], solution[indices[1]]
                interval_1 = np.random.choice([i for i in range(1, len(tour_1) - 1)], 2, replace=False)
                interval_2 = np.random.choice([i for i in range(1, len(tour_2) - 1)], 2, replace=False)
                start_1, start_2 = min(interval_1[0], interval_1[1]), min(interval_2[0], interval_2[1])
                end_1, end_2 = max(interval_1[0], interval_1[1]), max(interval_2[0], interval_2[1])
                tmp = copy.deepcopy(tour_1[start_1: end_1])
                tour_1[start_1: end_1] = reversed(copy.deepcopy(tour_2[start_2: end_2]))
                tour_2[start_2: end_2] = reversed(tmp)
            if self.solution_cost(solution) > (1 - lam) * self.solution_cost(optimal_solution):
                basic_solution = copy.deepcopy(solution)
        return basic_solution

    def cross_operator(self, solution_1, solution_2):
        solution = [[depot] for depot in self.depots]
        sol_1, sol_2 = copy.deepcopy(solution_1), copy.deepcopy(solution_2)
        sol_1.sort(key=lambda x: self.tour_length(x) / len(x) - 2 if len(x) > 2 else 100000)
        sol_2.sort(key=lambda x: self.tour_length(x) / len(x) - 2 if len(x) > 2 else 100000)
        rest_routes = []
        for _ in self.depots:
            pop_index = None
            route_1, route_2 = sol_1[0], sol_2[0]
            if self.tour_length(route_1) * (len(route_2) - 2) > self.tour_length(route_2) * (len(route_1) - 2):
                tmp = copy.deepcopy(sol_1)
                sol_1 = copy.deepcopy(sol_2)
                sol_2 = copy.deepcopy(tmp)
                route_1, route_2 = sol_1[0], sol_2[0]
            solution[np.where(self.depots == route_1[0])[0].item()] += copy.deepcopy(route_1[1:])
            for i in range(len(sol_2)):
                route = [sol_2[i][0]]
                for city in sol_2[i][1: -1]:
                    if city not in route_1:
                        route.append(copy.deepcopy(city))
                route.append(route[0])
                sol_2[i] = copy.deepcopy(route)
                if route[0] == sol_1[0][0]:
                    rest_routes.append(copy.deepcopy(sol_2[i]))
                    pop_index = i
            sol_1.pop(0)
            sol_2.pop(pop_index)
            for i in range(len(rest_routes)):
                route = [rest_routes[i][0]]
                for city in rest_routes[i][1: -1]:
                    if city not in route_1:
                        route.append(copy.deepcopy(city))
                route.append(route[0])
                rest_routes[i] = copy.deepcopy(route)
            sol_1.sort(key=lambda x: self.tour_length(x) / len(x) - 2 if len(x) > 2 else 100000)
            sol_2.sort(key=lambda x: self.tour_length(x) / len(x) - 2 if len(x) > 2 else 100000)
        for route in solution:
            if len(route) == 1:
                route.append(route[0])
        un_visited_cities = [city for route in rest_routes for city in route[1: -1]]
        for city in un_visited_cities:
            optimal, value = None, 100000
            for tour in solution:
                for i in range(1, len(tour)):
                    tour.insert(i, city)
                    if self.solution_cost(solution) < value:
                        value = self.solution_cost(solution)
                        optimal = copy.deepcopy(solution)
                    tour.pop(i)
            solution = copy.deepcopy(optimal) if optimal is not None else solution

        return solution

    def solution_cost(self, solution):
        if len(self.stations) == 0:
            return self.solution_length(solution)
        cost = 0
        for path in solution:
            route = [path[0]]
            energy = self.full
            next_customer = 1
            station_visited = [False for _ in self.stations]
            while route[-1] != path[-1] or len(route) < len(path):
                if energy > self.distance[route[-1]][path[next_customer]]:
                    energy -= self.distance[route[-1]][path[next_customer]]
                    cost += self.distance[route[-1]][path[next_customer]]
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
                    cost += min_distance
                    if closest_station is None:
                        return 1e10
                    route.append(self.stations[closest_station])
                    if energy < min_distance:
                        cost += self.alpha * (min_distance - energy)
                    energy = self.full
                    station_visited[closest_station] = True
        return cost

    def update_population(self, solution):
        cost, w = 0, None
        for i in range(self.size):
            if self.solution_cost(self.population[i]) > cost:
                cost = self.solution_cost(self.population[i])
                w = i
        dist, c = 100000, None
        for i in range(self.size):
            if self.hamming(solution, self.population[i]) < dist:
                dist = self.hamming(solution, self.population[i])
                c = i
        if self.solution_cost(solution) < self.solution_cost(self.population[c]) and dist < self.beta * len(
                self.cities):
            self.population.pop(c)
            self.population.append(copy.deepcopy(solution))
        elif self.solution_cost(solution) < cost and dist > self.beta * len(self.cities):
            self.population.pop(w)
            self.population.append(copy.deepcopy(solution))

    def hamming(self, solution_1, solution_2):
        dist = len(self.cities)
        for i in range(len(self.depots)):
            for city in solution_1[i][1: -1]:
                if city in solution_2[i]:
                    dist -= 1
        return dist

    def solve(self):
        for i in range(self.size):
            self.initial_population()
        for i in range(self.size):
            self.population[i] = self.variable_neighborhood_search(self.population[i])
        iteration, run_time = 0, 0
        while iteration < self.iters and run_time < self.run_time:
            iteration += 1
            count = time.time()
            indices = np.random.choice([i for i in range(self.size)], 2, replace=False)
            solution_1, solution_2 = self.population[indices[0]], self.population[indices[1]]
            solution = self.cross_operator(solution_1, solution_2)
            solution = self.variable_neighborhood_search(solution)
            self.update_population(solution)
            run_time += time.time() - count
        self.population.sort(key=lambda x: self.solution_cost(x))
        for route in self.population[0]:
            routes = self.build_routes([route])
            if len(routes) > 0:
                self.solution.append(sorted(routes, key=self.tour_length)[0])
            else:
                self.solution.append(route)
