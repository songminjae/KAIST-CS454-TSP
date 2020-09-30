import os
import sys
import math
import argparse
import numpy as np
import random


class TSP:
    def __init__(self):
        self.coordinates = []
        self.distances = []


def main():
    f, pop_size, fitness_limit, generation_limit, parent_number = handle_args()
    load_file(f)
    # 무작위로 뽑은 경로들의 집합 - <var_route>  [ [1, 2, 3, ...], [2, 3, 1, ...], ... ]
    var_route = init_pop(pop_size)
    iter = 0
    pop = var_route[:]
    while iter < 1:  # 원래는 generation_limit사용해야함
        iter += 1
        fit_list = eval_fit(pop)
        print(fit_list)
        parent_index = tournament_selection(fit_list, parent_number)
        print(parent_index)

        # we should do cross over with <pop> using <parent_index>


def load_file(tsp_file):
    global tsp
    with open(tsp_file, "r") as f:
        while True:
            l = f.readline()
            pl = l.strip().split()
            if(pl[0] == "NODE_COORD_SECTION"):
                break
        for line in f:
            l = line.strip()
            if(l == "EOF"):
                break
            city = l.split()
            tsp.coordinates.append(
                (int(city[0]), float(city[1]), float(city[2])))
        f.close()
        print("tsp.coordinate len:", len(tsp.coordinates))
        coordi = tsp.coordinates
        tsp.distances = [[math.hypot((coordi[i][1] - coordi[j][1]), (coordi[i][2] - coordi[j][2]))
                          for j in range(len(coordi))] for i in range(len(coordi))]
        # print(tsp.distances[2][2])
        # print(tsp.distances[1][2])  # 2번 도시랑 3번 도시 사이의 거리
        # print(math.hypot((1.54080e+04 - 1.19150e+04), (7.87600e+03 - 4.77400e+03)))

    # for i in range(len(tsp.coordinates)):
    #     for j in range(len(tsp.coordinates)):
    #         coordi = tsp.coordinates
    #         tsp.coordinates[[math.hypot(
    #             (coordi[i][1] - coordi[j][1]), (coordi[i][2] - coordi[j][2]))]]


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", "--file_name", default="rl11849.tsp")
    parser.add_argument("-p", "--population_size", default=200,
                        help="size of population", type=int)
    parser.add_argument("-f", "--fitness_evaluation", default=100,
                        help="total number of fitness evaluation limit", type=int)
    parser.add_argument("-g", "--generation_size",
                        default=1000, help="generation limit", type=int)
    parser.add_argument("-parent", "--parent_number",
                        default=2, help="number of parent", type=int)
    args = parser.parse_args()
    return args.file_name, args.population_size, args.fitness_evaluation, args.generation_size, args.parent_number


def make_solution(path):
    with open("solution.csv", "w") as f:
        for i in path:
            f.write(i + "/n")


def init_pop(pop_size):
    var_route = list()
    individual = list(range(len(tsp.coordinates)))  # 무작위 경로 하나
    # print(individual)
    # print(var_route)
    for i in range(pop_size):
        # random.shuffle(individual)
        # var_route.append(individual)
        shuffled = individual[:]
        random.shuffle(shuffled)
        var_route.append(shuffled)
    # print(var_route)
    return var_route


def eval_fit(route_list):
    print("evaluating fitness\n")
    fit_list = [0] * len(route_list)
    for i in range(len(route_list)):
        dist = cal_distance(route_list[i])
        fit_list[i] = dist

    return fit_list


def select_parent():
    print("select parent\n")


def create_offs():
    print("create offsprings\n")


def mutate_offs():
    print("mutate offsrpings\n")

##### tools for calculation ####


def cal_distance(route):  # route 는 도시들 경로 [4, 2, 1, ...]
    dist = 0
    for i in range(len(route)):
        next = 0
        if i != len(route) - 1:
            next = i + 1
        dist = dist + tsp.distances[route[i]][route[next]]
    return dist


# choose limit number of routes with better fitness using tournament selec
def tournament_selection(fit_list, limit):  # fit_list = [fit_1, fit_2, ...]
    result = list(range(len(fit_list)))
    while len(result) > limit:
        match_num = len(result) // 2
        selected = []
        for i in range(match_num):
            first = result[i * 2]
            second = result[i * 2 + 1]
            print("first", first)
            print("second", second)
            print(result)
            if fit_list[first] > fit_list[second]:
                selected.append(first)
            else:
                selected.append(second)
        if len(result) % 2 == 1:
            selected.append(result[len(result) - 1])

        result = selected[:]

    return result


if __name__ == '__main__':
    tsp = TSP()
    main()
