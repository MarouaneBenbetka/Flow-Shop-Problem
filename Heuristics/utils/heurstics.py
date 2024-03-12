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
