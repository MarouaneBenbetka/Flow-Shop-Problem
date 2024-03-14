import numpy as np
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import calculate_makespan


# NEH

def _order_jobs_in_descending_order_of_total_completion_time(processing_times):
    total_completion_time = processing_times.sum(axis=1)
    return np.argsort(total_completion_time, axis=0).tolist()


def _insertion(sequence, position, value):
    new_seq = sequence[:]
    new_seq.insert(position, value)
    return new_seq


def neh_algorithm(processing_times):
    ordered_sequence = _order_jobs_in_descending_order_of_total_completion_time(
        processing_times)
    # Define the initial order
    J1, J2 = ordered_sequence[:2]
    sequence = [J1, J2] if calculate_makespan(processing_times, [
        J1, J2]) < calculate_makespan(processing_times, [J2, J1]) else [J2, J1]
    del ordered_sequence[:2]
    # Add remaining jobs
    for job in ordered_sequence:
        Cmax = float('inf')
        best_sequence = []
        for i in range(len(sequence)+1):
            new_sequence = _insertion(sequence, i, job)
            Cmax_eval = calculate_makespan(processing_times, new_sequence)
            if Cmax_eval < Cmax:
                Cmax = Cmax_eval
                best_sequence = new_sequence
        sequence = best_sequence
    return sequence, Cmax


# ---------------------------------#


# HAM

def _indices_from_list_values(master: np.ndarray, values: list) -> list:
    indexes = []
    for element in values:
        for i in range(master.size):
            if master[i] == element and i not in indexes:
                indexes.append(i)
                continue
    return indexes


def _ham_sol_1(Pi1: np.ndarray, Pi2: np.ndarray) -> list:
    diff = Pi2 - Pi1
    sol = np.argsort(diff, axis=0).tolist()
    sol.reverse()       # in decreasing order
    return sol


def _ham_sol_2(Pi1: np.ndarray, Pi2: np.ndarray) -> list:

    diff = Pi2 - Pi1
    according_pi1 = np.argwhere(diff >= 0)
    according_pi2 = np.argwhere(diff < 0)

    # Order Pi1 in increasing order
    Pi1_sorted = np.sort(Pi1[according_pi1], axis=None).tolist()
    Pi1_list = _indices_from_list_values(Pi1, Pi1_sorted)

    # Order Pi2 in decreasing order
    Pi2_sorted = np.sort(Pi2[according_pi2], axis=None).tolist()
    Pi2_sorted.reverse()
    Pi2_list = _indices_from_list_values(Pi2, Pi2_sorted)

    return Pi1_list + Pi2_list


def ham_heuristic(processing_times: np.ndarray) -> list:
    _, m = processing_times.shape
    left = processing_times[:, :int(m/2)]
    right = processing_times[:, int(m/2):]

    Pi1 = left.sum(axis=1)
    Pi2 = right.sum(axis=1)

    solution1 = _ham_sol_1(Pi1, Pi2)
    solution2 = _ham_sol_2(Pi1, Pi2)
    Cmax1 = calculate_makespan(processing_times, solution1)
    Cmax2 = calculate_makespan(processing_times, solution2)

    if Cmax1 < Cmax2:
        return solution1, Cmax1
    else:
        return solution2, Cmax2


# ---------------------------------#

# PALMER


def _init_weights(nb_machines):
    lst = np.array([(2*i - 1 - nb_machines) for i in range(nb_machines)])
    return lst - np.mean(lst)


def _compute_weighted_sum(dist_mat, weights, nb_jobs, nb_machines):
    # Adjusted to work with dist_mat having dimensions [nb_jobs, nb_machines]
    weighted_sum = []
    for i in range(nb_jobs):
        somme = np.dot(dist_mat[i, :], weights)
        weighted_sum.append(somme)
    return np.array(weighted_sum)


def _update_c(dist_mat, c, ordre, nb_machines, nb_jobs):
    # Adjusted for dist_mat with dimensions [nb_jobs, nb_machines]
    c[0][0] = dist_mat[ordre[0], 0]
    for j in range(1, nb_machines):
        c[0][j] = dist_mat[ordre[0], j] + c[0][j-1]
    for i in range(1, nb_jobs):
        c[i][0] = dist_mat[ordre[i], 0] + c[i-1][0]
    for i in range(1, nb_jobs):
        for j in range(1, nb_machines):
            c[i][j] = max(c[i-1][j], c[i][j-1]) + dist_mat[ordre[i], j]
    return c


def _get_cmax(c, ordre_opt, nb_machines):
    return c[len(ordre_opt)-1][nb_machines-1]


def _lower_bound(dist_mat, nb_machines, nb_jobs):
    # Adjusted for dist_mat with dimensions [nb_jobs, nb_machines]
    lb = []
    for i in range(nb_machines):
        a = dist_mat.sum(axis=0)
        if i == 0:
            bound = a[i] + min(dist_mat[:, 1:].sum(axis=0))
        elif i == (nb_machines - 1):
            bound = a[i] + min(dist_mat[:, :i].sum(axis=0))
        else:
            bound = a[i] + min(dist_mat[:, :i].sum(axis=0)
                               ) + min(dist_mat[:, i+1:])
        lb.append(bound)
    return max(lb)


def run_palmer(dist_mat):
    nb_jobs, nb_machines = dist_mat.shape
    weights = _init_weights(nb_machines)

    weighted_sum = _compute_weighted_sum(
        dist_mat, weights, nb_jobs, nb_machines)
    ordre_opt = np.argsort(weighted_sum)[::-1]

    makespan = calculate_makespan(dist_mat, ordre_opt)

    return ordre_opt, makespan


# ---------------------------------#

# PMSKE


def _AVG(processing_times):
    return np.mean(processing_times, axis=1)


def _STD(processing_times):
    return np.std(processing_times, axis=1, ddof=1)


def _skewness_SKE(processing_times):
    """
    Calculates the skewness of job processing times across machines.
    Return : Skewness values for each job

    """
    num_jobs, num_machines = processing_times.shape
    skewness_values = []

    for i in range(num_jobs):
        avg_processing_time = np.mean(processing_times[i, :])
        numerateur = 0
        denominateur = 0

        for j in range(num_machines):
            som = (processing_times[i, j] - avg_processing_time)
            numerateur += som ** 3
            denominateur += som ** 2

        numerateur *= (1 / num_machines)
        denominateur = (np.sqrt(denominateur * (1 / num_machines))) ** 3

        skewness_values.append(numerateur / denominateur)

    return np.array(skewness_values)


def PRSKE(processing_times):
    """
    Calculates the job sequence based on the PRSKE priority rule

    """
    avg = _AVG(processing_times)   # Calculate average processing times

    # Calculate standard deviation processing times
    std = _STD(processing_times)

    skw = _skewness_SKE(processing_times)  # Calculate Skewness

    order = skw + std + avg

    # Sort in descending order
    sorted_order = sorted(
        zip(order, list(range(processing_times.shape[0]))), reverse=True)

    sequence = [job for _, job in sorted_order]

    makespan = calculate_makespan(processing_times, sequence)

    return sequence,  makespan


# ---------------------------------#

# GUPTA

def _min_sum_processing(job_index, processing_times):
    min_sum = np.inf
    for i in range(processing_times.shape[1] - 1):
        sum_for_pair = processing_times[job_index,
                                        i] + processing_times[job_index, i + 1]
        if sum_for_pair < min_sum:
            min_sum = sum_for_pair
    return min_sum


def _calculate_priority(job_index, processing_times):
    diff = float(processing_times[job_index, 0] -
                 processing_times[job_index, -1])
    sign = (diff > 0) - (diff < 0)
    return sign / _min_sum_processing(job_index, processing_times)


def gupta_heuristic(processing_times):
    priorities = [_calculate_priority(i, processing_times)
                  for i in range(processing_times.shape[0])]
    total_times = [np.sum(processing_times[i])
                   for i in range(processing_times.shape[0])]
    sequence = sorted(range(len(priorities)), key=lambda k: (
        priorities[k], total_times[k]))
    return sequence, calculate_makespan(processing_times, sequence)


# Special

def _johnsons_rule(machine1, machine2):
    artificial_jobs = list(zip(machine1, machine2))

    jobs_sorted = sorted(enumerate(artificial_jobs), key=lambda x: min(x[1]))
    U = [job for job in jobs_sorted if job[1][0] < job[1][1]]
    V = [job for job in jobs_sorted if job[1][0] >= job[1][1]]

    sequence = [job[0] for job in U] + [job[0] for job in reversed(V)]
    return sequence


def special_heuristic(processing_times):
    _, n_machines = processing_times.shape
    best_sequence = None
    best_makespan = float('inf')

    for k in range(1, n_machines - 1):
        weights_front = np.array([n_machines - i for i in range(k)])
        weights_back = np.array([i + 1 for i in range(k, n_machines)])
        AM1 = processing_times[:, :k].dot(weights_front)
        AM2 = processing_times[:, k:].dot(weights_back)

        sequence = _johnsons_rule(AM1, AM2)
        makespan = calculate_makespan(processing_times, sequence)
        if makespan < best_makespan:
            best_makespan = makespan
            best_sequence = sequence

    return best_sequence, makespan


# ---------------------------------#

# CDS
def _johnson_method(processing_times):
    jobs, machines = processing_times.shape
    copy_processing_times = processing_times.copy()
    maximum = processing_times.max() + 1
    m1 = []
    m2 = []

    if machines != 2:
        raise Exception("Johson method only works with two machines")

    for i in range(jobs):
        minimum = copy_processing_times.min()
        position = np.where(copy_processing_times == minimum)

        if position[1][0] == 0:
            m1.append(position[0][0])
        else:
            m2.insert(0, position[0][0])

        copy_processing_times[position[0][0]] = maximum

    return m1+m2


# python code for CDS heuristic

def cds_heuristic(processing_times):
    processing_times = processing_times.T
    nb_machines, nb_jobs = processing_times.shape

    best_cost = math.inf

    machine_1_times = np.zeros((nb_jobs, 1))
    machine_2_times = np.zeros((nb_jobs, 1))

    # iterate through the nb_machines-1 auxiliary n-job 2-machines problems

    for k in range(nb_machines - 1):
        machine_1_times[:, 0] += processing_times[:][k]
        machine_2_times[:, 0] += processing_times[:][-k-1]

        jn_times = np.concatenate((machine_1_times, machine_2_times), axis=1)
        seq = _johnson_method(jn_times)
        cost = calculate_makespan(jn_times, seq)
        if cost < best_cost:
            best_cost = cost
            best_seq = seq

    return best_seq, calculate_makespan(processing_times.T, best_seq)


# ---------------------------------#

# NRH

def NRH(processing_times, shuffle_count=10):
    transformed = np.vectorize(lambda row, col: processing_times[row, col]/(
        np.exp(-col*1.6)))(*np.indices(processing_times.shape))

    # sum for each job (sum each row elements)
    transformed_sum = np.sum(transformed, axis=1)
    transformed_reshaped = transformed_sum.reshape(-1)

    initial_order = list(sorted(range(
        processing_times.shape[0]), key=lambda x: transformed_reshaped[x], reverse=True))

    current_make_span = calculate_makespan(processing_times, initial_order)
    current_order = initial_order

    # for i in range(sxorder = list(copy)

    return current_order, current_make_span

# ---------------------------------#


# CHEN

def chen_heuristic(processing_times):

    num_jobs = processing_times.shape[0]

    # Calcule de la somme de temps opératoires S(i) pour chaque tâche i
    sum_processing_times = [sum(processing_times[i]) for i in range(num_jobs)]

    job_max_sum = max(sum_processing_times)
    job_c = sum_processing_times.index(job_max_sum)

    remaining_jobs = [i for i in range(num_jobs) if i != job_c]

    sorted_jobs_le = sorted(
        remaining_jobs, key=lambda i: processing_times[i][0])

    sorted_jobs_gt = sorted(
        remaining_jobs, key=lambda i: processing_times[i][-1], reverse=True)

    S_a = [i for i in sorted_jobs_le if processing_times[i]
           [0] <= processing_times[i][-1]]
    S_b = [i for i in sorted_jobs_gt if processing_times[i]
           [0] > processing_times[i][-1]]

    sequence = S_a + [job_c] + S_b

    makespan = calculate_makespan(processing_times, sequence)

    return sequence, makespan
