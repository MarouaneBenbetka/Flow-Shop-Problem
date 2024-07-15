

import numpy as np
import random
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args


from utils.utils import calculate_makespan


def swap(solution, i, k):
    sol = solution.copy()
    sol[i], sol[k] = sol[k], sol[i]
    return sol


def random_swap(solution, processing_times):
    i = random.choice(list(solution))
    j = random.choice(list(solution))

    while i == j:
        j = random.choice(list(solution))

    new_solution = swap(solution, i, j)

    return new_solution, calculate_makespan(processing_times, new_solution)


def best_swap(solution, processing_times):
    sequence = solution.copy()
    num_jobs = len(solution)
    Cmax = calculate_makespan(processing_times, solution)

    for i in range(num_jobs):
        for j in range(i+1, num_jobs):
            new_solution = swap(sequence, i, j)
            makespan = calculate_makespan(processing_times, new_solution)

            if makespan < Cmax:
                sequence = new_solution
                Cmax = makespan

    return sequence, Cmax


def first_admissible_swap(solution, processing_times):

    num_jobs = len(solution)
    Cmax = calculate_makespan(processing_times, solution)

    for i in range(num_jobs):
        for j in range(i+1, num_jobs):
            new_solution = swap(solution, i, j)
            makespan = calculate_makespan(processing_times, new_solution)

            if makespan < Cmax:
                return new_solution, makespan

    return solution, Cmax


def fba_swap(solution, processing_times, best_global_sol):
    sequence = solution.copy()
    num_jobs = len(sequence)
    Cmax = calculate_makespan(processing_times, sequence)
    Smax = calculate_makespan(processing_times, best_global_sol)
    for i in range(num_jobs):
        for j in range(i+1, num_jobs):
            new_solution = swap(solution, i, j)
            makespan = calculate_makespan(processing_times, new_solution)

            # First improving solution
            if makespan < Cmax:
                # Improves the global solution
                if makespan < Smax:
                    return new_solution, makespan, new_solution
                Cmax = makespan
                sequence = new_solution

    return sequence, Cmax, best_global_sol


def get_neighbor(processing_times, solution, method='random_swap'):
    if method == 'random_swap':
        sol, val = random_swap(solution, processing_times)
    elif method == 'best_swap':
        sol, val = best_swap(solution, processing_times)
    elif method == 'first_admissible_swap':
        sol, val = first_admissible_swap(solution, processing_times)
    else:
        i = random.randint(0, 2)
        if i == 0:
            sol, val = random_swap(solution, processing_times)
        elif i == 1:
            sol, val = best_swap(solution, processing_times)
        elif i == 2:
            sol, val = first_admissible_swap(solution, processing_times)
    return sol, val


def RS(processing_times, initial_solution, temp, method='random_swap', alpha=0.6, nb_palier=10, it_max=100):

    start_time = time.time()
    solution = initial_solution.copy()
    makespan = calculate_makespan(processing_times, solution)
    print('init_sol: ', solution, ' makespan = ', makespan, "\n")
    it = 0
    while it < it_max:
        for i in range(nb_palier):
            sol, value = get_neighbor(processing_times, solution, method)
            # print('Swap_sol: ',sol,' makespan = ', value)
            delta = makespan - value
            if delta > 0:
                solution = sol
                makespan = value
            else:
                if random.uniform(0, 1) < math.exp(delta / temp):
                    solution = sol
        temp = alpha * temp
        it += 1
    print("Elapsed time:", time.time()-start_time, "seconds")
    return solution


def RS_fba(processing_times, initial_solution, temp, alpha=0.6, nb_palier=1, it_max=100):

    intitial_global = np.random.permutation(len(processing_times)).tolist()

    start_time = time.time()
    solution = initial_solution.copy()
    makespan = calculate_makespan(processing_times, solution)
    print('init_sol: ', solution, ' makespan = ', makespan)
    it = 0
    print('initial_global_solution: ', intitial_global, ' global_makespan = ',
          calculate_makespan(processing_times, intitial_global), "\n")
    best_global = intitial_global.copy()
    while it < it_max:
        for i in range(nb_palier):
            sol, value, best_global = fba_swap(
                solution, processing_times, best_global)
            # print('FBA_swap_sol: ',sol,' makespan = ', value)
            delta = makespan - value
            if delta > 0:
                solution = sol
                makespan = value
            else:
                if random.uniform(0, 1) < math.exp(delta / temp):
                    solution = sol
        temp = alpha * temp
        it += 1
    print("Elapsed time:", time.time()-start_time, "seconds")
    return solution
