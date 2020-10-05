import os
import sys
import math
import argparse
import random


class TSP:
    def __init__(self):
        self.coordinates = []
        self.counter = 0


def main():
    f, pop_size, fitness_limit, generation_limit, parent_number, elite_number, mutation_rate, neighbor_size, crossover_method, padding_number, improve_limit, is_greedy, is_memetic = handle_args()
    load_file(f)
    # 무작위로 뽑은 경로들의 집합 - <var_route>  [ [1, 2, 3, ...], [2, 3, 1, ...], ... ]
    ## initial population ##
    var_route, var_route_fit = init_pop(pop_size, padding_number, is_greedy)
    print("init pop's fitness", var_route_fit)
    ######################for memetic algorithm##########################
    if is_memetic == 0:
        local_route, local_route_fit = local_search(
            var_route, var_route_fit, neighbor_size, fitness_limit, improve_limit)
    else:
        local_route = var_route[:]
        local_route_fit = var_route_fit[:]
    #######################memetic algorithm#############################
    print("local fitness", local_route_fit)
    iter = 0
    pop = local_route[:]
    while iter <= generation_limit:  # 원래는 generation_limit사용해야함
        iter += 1
        print("this is generation #", iter)
        ######################for memetic algorithm###################
        if is_memetic == 0:
            ## evaluate fitness ##
            fit_list = eval_fit(pop)
        else:
            using_pop = pop[:]
            pop, fit_list = local_search(using_pop, eval_fit(
                using_pop), neighbor_size, fitness_limit, improve_limit)
        print(fit_list)
        ## select as parent ##
        parent_index = tournament_selection(fit_list, parent_number)
        if iter == generation_limit:
            print("almost there")
            parent_index = tournament_selection(fit_list, 1)
            break
        # we should do cross over with <pop> using <parent_index>
        ## create offs ##
        parent = pop[:]
        pop = []
        # print("original_parent_len", len(parent))
        print("crossover with method", crossover_method)
        for i in range(len(parent)):
            parent_list = random.sample(parent_index, 2)
            if crossover_method == 2:
                child = CX(
                    parent[parent_list[0]], parent[parent_list[1]])
                pop.append(child)
                child = CX(
                    parent[parent_list[1]], parent[parent_list[0]])
                pop.append(child)
            elif crossover_method == 1:
                child = PMCrossover(
                    parent[parent_list[0]], parent[parent_list[1]])
                pop.append(child)
            else:
                child = crossover(
                    parent[parent_list[0]], parent[parent_list[1]])
                pop.append(child)
        # elite_number 만큼 pop에 추가해줘야함
        ## mutate pop(children) ##
        semi_mutate_result = mutate_offs(pop, mutation_rate)
        child_fit_list = eval_fit(semi_mutate_result)
        child_best_index = tournament_selection(
            child_fit_list, pop_size - elite_number)
        mutate_result = []
        for x in child_best_index:
            mutate_result.append(semi_mutate_result[x])
        ## add elites ##
        elite_index = tournament_selection(fit_list, elite_number)
        for index in elite_index:
            mutate_result.append(parent[index])
        pop = mutate_result[:]

    final_index = parent_index[0]
    make_solution(pop[final_index])
    print("beast dist", fit_list[final_index])


def get_neighbors(route, neighbor_size):
    neighbors = []
    for i in range(neighbor_size):
        x, y = [random.choice(range(0, len(route) - 1)),
                random.choice(range(0, len(route) - 1))]
        if x == y:
            continue
        high = max(x, y)
        low = min(x, y)
        near = route[:low] + route[low: high][::-1] + route[high:]
        neighbors.append(near)
    return neighbors


def local_search(var_route, var_route_fit, neighbor_size, fitness_limit, improve_limit):
    print("local search for init_pops")
    optimas = []
    optimas_fit = []
    for i in range(len(var_route)):
        # get neighbours
        route = var_route[i]
        route_fit = var_route_fit[i]
        count = 0
        while True:
            pre_best_fit = route_fit
            local_best = route
            local_best_fit = route_fit
            neighbor = get_neighbors(local_best, neighbor_size)
            for n in neighbor:
                dist = distance(n)
                if local_best_fit > dist:
                    local_best = n
                    local_best_fit = dist
            route = local_best
            route_fit = local_best_fit
            count += 1
            if count > fitness_limit:
                if test_for_break(route_fit, pre_best_fit, improve_limit):
                    break
        optimas.append(route)
        optimas_fit.append(route_fit)
    return optimas, optimas_fit


def test_for_break(route_fit, pre_fit, improve_limit):
    improve = (pre_fit - route_fit) / pre_fit
    if improve < improve_limit:
        print("improved", improve, "pre_fit:", pre_fit, "this_fit:", route_fit)
        return True
    return False


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
        coordi = tsp.coordinates


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", "--file_name", default="rl11849.tsp")
    parser.add_argument("-p", "--population_size", default=5,
                        help="size of population", type=int)
    parser.add_argument("-l", "--local_fitness_evaluation", default=2000,
                        help=" fitness evaluation limit for local search!!", type=int)
    parser.add_argument("-g", "--generation_size",
                        default=200, help="generation limit", type=int)
    parser.add_argument("-parent", "--parent_number",
                        default=2, help="number of parent", type=int)
    parser.add_argument("-e", "--elite_number",
                        default=3, help="elite number should not bigger than parent", type=int)
    parser.add_argument("-m", "--mutation_rate",
                        default=0.0001, help="mutation rate 11849 * rate", type=float)
    parser.add_argument("-n", "--neighbor_size",
                        default=50, help="neighbor size", type=int)
    parser.add_argument("-c", "--crossover_method",
                        default=0, help="0 for normal, 1 TMC, 2 CX ", type=int)
    parser.add_argument("-padd", "--padding_number",
                        default=0, help="padding for init", type=int)
    parser.add_argument("-i", "--improve_limit",
                        default=0.0001, help="limit for improvement", type=float)
    parser.add_argument("-greedy", "--is_greedy",
                        default=1, help="0 for not using greedy, 1 for greedy", type=int)
    parser.add_argument("-memetic", "--is_memetic",
                        default=1, help="0 for not memetic local search at init pop, 1 for memetic local search at each loop", type=int)
    args = parser.parse_args()
    return args.file_name, args.population_size, args.local_fitness_evaluation, args.generation_size, args.parent_number, args.elite_number, args.mutation_rate, args.neighbor_size, args.crossover_method, args.padding_number, args.improve_limit, args.is_greedy, args.is_memetic


def make_solution(path):
    with open("solution.csv", "w") as f:
        for i in path:
            f.write(str(i + 1) + "\n")


def init_pop(pop_size, padding_number, is_greedy):  # is_greedy 가 1이면 greedy, 0이면 x
    var_route = list()
    var_route_fit = []
    individual = list(range(len(tsp.coordinates)))  # 무작위 경로 하나
    for i in range(pop_size):
        # random.shuffle(individual)
        # var_route.append(individual)
        if is_greedy == 0:
            shuffled = individual[:]
            random.shuffle(shuffled)
        else:
            shuffled = make_greedy_route()
        ##better than 100 ?? ##
        best = shuffled[:]
        trial = shuffled[:]
        best_fit = distance(shuffled)
        for i in range(padding_number):
            if is_greedy == 0:
                random.shuffle(trial)
            else:
                trial = make_greedy_route()
            d = distance(trial)
            if(best_fit > d):
                best_fit = d
                best = trial[:]

        var_route.append(best)
        var_route_fit.append(best_fit)
        print("init_pop's fit", best_fit)
    return var_route, var_route_fit


def make_greedy_route():
    random_start = random.randint(0, len(tsp.coordinates) - 1)
    greedy_route = [random_start]
    for i in range(len(tsp.coordinates) - 1):
        start = greedy_route[len(greedy_route)-1]
        dist = 99999999
        greedy_index = 0
        for j in range(len(tsp.coordinates)):
            search_dist = distance_between(start, tsp.coordinates[j][0] - 1)
            if search_dist < dist:
                if j not in greedy_route:
                    dist = search_dist
                    greedy_index = j
        greedy_route.append(greedy_index)
    return greedy_route


def eval_fit(route_list):
    tsp.counter += 1
    print("evaluating fitness")
    fit_list = [0] * len(route_list)
    for i in range(len(route_list)):
        dist = distance(route_list[i])

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


def PMCrossover(r1, r2):
    c1, c2 = random.randint(0, len(r1) - 1), random.randint(0, len(r1) - 1)
    while c1 == c2:
        c2 = random.randint(0, len(r1) - 1)
    high = max(c1, c2)
    low = min(c1, c2)
    p1 = r1[: low]
    p2 = r1[high:]
    t1, t2 = r1[low: high], r2[low:high]
    new_p1 = []
    for c in p1:
        if c in t2:
            while c in t2:
                i = t2.index(c)
                c = t1[i]
        new_p1.append(c)
    new_p2 = []
    for c in p2:
        if c in t2:
            while c in t2:
                i = t2.index(c)
                c = t1[i]
        new_p2.append(c)
    return new_p1 + t2 + new_p2


def CX(r1, r2):
    first = r1[0]
    second = r1[len(r1)-1]
    index_1, index_2 = r2.index(first), r2.index(second)
    if(index_2 < index_1):
        r2 = r2[::-1]
        index_1, index_2 = r2.index(first), r2.index(second)

    loop_1 = r2[index_1:index_2] + r2[index_2:] + r2[:index_1]
    loop_2 = r2[:index_1][::-1] + \
        r2[index_2:][::-1]+r2[index_1:index_2][::-1]
    last = loop_2.pop()
    loop_2.insert(0, last)
    semi_loop_1 = loop_1[loop_1.index(first):loop_1.index(second)+1]
    semi_loop_2 = loop_2[loop_2.index(first):loop_2.index(second)+1]
    ##
    index_semi_loop_1 = []
    index_semi_loop_2 = []
    for x in semi_loop_1:
        index_semi_loop_1.append(r1.index(x))

    for x in semi_loop_2:
        index_semi_loop_2.append(r1.index(x))
    index_semi_loop_1.pop()
    index_semi_loop_2.pop()
    index_semi_loop_1.remove(0)
    index_semi_loop_2.remove(0)

    result1 = []
    for i in range(len(index_semi_loop_1)):
        using_best = index_semi_loop_1[i]
        best_set = [index_semi_loop_1[i]]
        using_best_set = [index_semi_loop_1[i]]
        for j in range(len(index_semi_loop_1)):
            if i < j:
                if using_best < index_semi_loop_1[j]:
                    using_best_set.append(index_semi_loop_1[j])
                    using_best = index_semi_loop_1[j]
            if len(best_set) < len(using_best_set):
                best_set = using_best_set[:]
        if len(result1) < len(best_set):
            result1 = best_set[:]

    result2 = []
    for i in range(len(index_semi_loop_2)):
        using_best = index_semi_loop_2[i]
        best_set = [index_semi_loop_2[i]]
        using_best_set = [index_semi_loop_2[i]]
        for j in range(len(index_semi_loop_2)):
            if i < j:
                if using_best < index_semi_loop_2[j]:
                    using_best_set.append(index_semi_loop_2[j])
                    using_best = index_semi_loop_2[j]
            if len(best_set) < len(using_best_set):
                best_set = using_best_set[:]
        if len(result2) < len(best_set):
            result2 = best_set[:]

    if len(result1) < len(result2):
        result = result2[:]
    else:
        result = result1[:]
    final_result = []
    for x in result:
        final_result.append(r1[x])
    real_final_result = [first] + final_result + [second]

    not_selected = []

    for x in r2:
        if x not in real_final_result:
            not_selected.append(x)

    cross = []
    for i in range(len(r1)):
        if r1[i] in real_final_result:
            cross.append(r1[i])
        else:
            put = not_selected.pop()
            cross.append(put)
    return cross


# target_list 는 [ [1, 2, 3 ...] , [2, 1, 3 ...] ...]
def mutate_offs(target_list, rate):
    print("mutate offsrpings\n")
    change_gene_num = int(rate * len(target_list[0]))
    result = target_list[:]
    for g in target_list:
        for i in range(change_gene_num):
            switch = random.sample(g, 2)
            s1_index = g.index(switch[0])
            s2_index = g.index(switch[1])
            g[s1_index] = switch[1]
            g[s2_index] = switch[0]

        result.append(g)
    return result


##### tools for calculation ####


def distance(route):
    dist = 0
    for i in range(len(route)):
        next = 0
        if i != len(route) - 1:
            next = i + 1
        coordi = tsp.coordinates
        # print(coordi[route[i]])
        dist = dist + \
            math.hypot((coordi[route[i]][1] - coordi[route[next]][1]),
                       (coordi[route[i]][2] - coordi[route[next]][2]))
    return dist


def distance_between(a, b):
    coordi = tsp.coordinates
    dist = math.hypot(coordi[a][1] - coordi[b][1],
                      (coordi[a][2] - coordi[b][2]))
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
            if fit_list[first] > fit_list[second]:
                selected.append(second)
            else:
                selected.append(first)
        if len(result) % 2 == 1:
            selected.append(result[len(result) - 1])
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

    return result


if __name__ == '__main__':
    tsp = TSP()
    main()
    print("total call of eval_fit function is:", tsp.counter)
