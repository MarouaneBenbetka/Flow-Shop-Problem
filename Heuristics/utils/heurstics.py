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
