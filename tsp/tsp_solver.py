import os
import sys
import math
import argparse
import random


class TSP:
    def __init__(self):
        self.coordinates = []
        # self.distances = []


def main():
    f, pop_size, fitness_limit, generation_limit, parent_number, elite_number, mutation_rate = handle_args()
    load_file(f)
    # 무작위로 뽑은 경로들의 집합 - <var_route>  [ [1, 2, 3, ...], [2, 3, 1, ...], ... ]
    ## initial population ##
    var_route = init_pop(pop_size)
    iter = 0
    pop = var_route[:]
    # print("1", pop[0][100])
    # print("2", pop[1][100])
    while iter <= generation_limit:  # 원래는 generation_limit사용해야함
        iter += 1
        print("this is generation #", iter)
        ## evaluate fitness ##
        fit_list = eval_fit(pop)
        print(fit_list)
        ## select as parent ##
        parent_index = tournament_selection(fit_list, parent_number)
        # print("this should be 10", len(parent_index))
        if iter == generation_limit:
            print("almost there")
            parent_index = tournament_selection(fit_list, 1)
            break
        # we should do cross over with <pop> using <parent_index>
        ## create offs ##
        parent = pop[:]
        pop = []
        # print("original_parent_len", len(parent))
        for i in range(len(parent)):
            parent_list = random.sample(parent_index, 2)
            # print("hh", parent_list)
            child = crossover(
                parent[parent_list[0]], parent[parent_list[1]])
            pop.append(child)
        # print("offs len 92", len(pop))
        # elite_number 만큼 pop에 추가해줘야함
        ## mutate pop(children) ##
        semi_mutate_result = mutate_offs(pop, mutation_rate)
        child_fit_list = eval_fit(semi_mutate_result)
        # print(len(child_fit_list), "should be 100")
        child_best_index = tournament_selection(child_fit_list, 20)
        mutate_result = []
        for x in child_best_index:
            mutate_result.append(semi_mutate_result[x])
        # print(len(mutate_result), "this should 20")
        ## add elites ##
        # print("fit_len 100", len(fit_list))
        elite_index = tournament_selection(fit_list, 80)
        # print("this is elite", elite_index)
        # print("elite_len 80", len(elite_index))
        for index in elite_index:
            mutate_result.append(parent[index])
        # print("this should be 100", len(mutate_result))
        pop = mutate_result[:]

    final_index = parent_index[0]
    make_solution(pop[final_index])
    print("beast dist", fit_list[final_index])


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
                (int(city[0]), round(float(city[1]), 3), round(float(city[2]), 3)))
        f.close()
        # print("tsp.coordinate len:", len(tsp.coordinates))
        coordi = tsp.coordinates
        # tsp.distances = [[math.hypot((coordi[i][1] - coordi[j][1]), (coordi[i][2] - coordi[j][2]))
        #                   for j in range(len(coordi))] for i in range(len(coordi))]


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", "--file_name", default="rl11849.tsp")
    parser.add_argument("-p", "--population_size", default=100,
                        help="size of population", type=int)
    parser.add_argument("-f", "--fitness_evaluation", default=100,
                        help="total number of fitness evaluation limit", type=int)
    parser.add_argument("-g", "--generation_size",
                        default=100, help="generation limit", type=int)
    parser.add_argument("-parent", "--parent_number",
                        default=10, help="number of parent", type=int)
    parser.add_argument("-e", "--elite_number",
                        default=8, help="elite number should not bigger than parent", type=int)
    parser.add_argument("-m", "--mutation_rate",
                        default=0.4, help="mutation rate", type=float)
    args = parser.parse_args()
    return args.file_name, args.population_size, args.fitness_evaluation, args.generation_size, args.parent_number, args.elite_number, args.mutation_rate


def make_solution(path):
    with open("solution.csv", "w") as f:
        for i in path:
            f.write(str(i + 1) + "\n")


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
    print("evaluating fitness")
    fit_list = [0] * len(route_list)
    for i in range(len(route_list)):
        dist = distance(route_list[i])
        # print(dist)
        fit_list[i] = dist
    return fit_list


def crossover(r1, r2):
    c1, c2 = random.randint(0, len(r1) - 1), random.randint(0, len(r1) - 1)
    while c1 == c2:
        c2 = random.randint(0, len(r1) - 1)

    sub_r1 = r1[min(c1, c2): max(c1, c2) + 1]
    sub_r2 = []
    for city in r2:
        if city not in sub_r1:
            sub_r2.append(city)
    return sub_r1 + sub_r2


# target_list 는 [ [1, 2, 3 ...] , [2, 1, 3 ...] ...]
def mutate_offs(target_list, rate):
    print("mutate offsrpings\n")
    change_gene_num = int(rate * len(target_list[0]))
    result = []
    # print("change_gene_num", change_gene_num)
    for g in target_list:
        # print("this should be 11849", len(g))
        for i in range(change_gene_num):
            switch = random.sample(g, 2)
            # print(switch)
            s1_index = g.index(switch[0])
            s2_index = g.index(switch[1])
            g[s1_index] = switch[1]
            g[s2_index] = switch[0]

        result.append(g)
    return result


##### tools for calculation ####


# def cal_distance(route):  # route 는 도시들 경로 [4, 2, 1, ...]
#     dist = 0
#     for i in range(len(route)):
#         next = 0
#         if i != len(route) - 1:
#             next = i + 1
#         dist = dist + tsp.distances[route[i]][route[next]]
#     return dist


def distance(route):
    dist = 0
    for i in range(len(route)):
        next = 0
        if i != len(route) - 1:
            next = i + 1
        coordi = tsp.coordinates
        dist = dist + \
            math.hypot((coordi[route[i]][1] - coordi[route[next]][1]),
                       (coordi[route[i]][2] - coordi[route[next]][2]))
    return dist


# choose limit number of routes with better fitness using tournament selec
def tournament_selection(fit_list, limit):  # fit_list = [fit_1, fit_2, ...]
    result = list(range(len(fit_list)))
    if len(fit_list) // limit == 1:
        early_return = []
        for i in range(len(result)):
            early_return.append((result[i], fit_list[result[i]]))
        early_return.sort(key=lambda x: x[1])
        while len(early_return) != limit:
            early_return.pop()
        early = []
        for i in range(len(early_return)):
            early.append(early_return[i][0])
        return early

    while len(result) > limit:
        match_num = len(result) // 2
        selected = []
        for i in range(match_num):
            first = result[i * 2]
            second = result[i * 2 + 1]
            # print("first", first)
            # print("second", second)
            # print(result)
            if fit_list[first] > fit_list[second]:
                selected.append(second)
            else:
                selected.append(first)
        if len(result) % 2 == 1:
            selected.append(result[len(result) - 1])
        # print(selected)
        if len(selected) // limit == 1:
            copy = []
            for i in range(len(selected)):
                copy.append((selected[i], fit_list[selected[i]]))
            copy.sort(key=lambda x: x[1])
            while len(copy) != limit:
                copy.pop()
            selected = []
            for i in range(len(copy)):
                selected.append(copy[i][0])

        result = selected[:]
    # print(result)

    return result


if __name__ == '__main__':
    tsp = TSP()
    main()
