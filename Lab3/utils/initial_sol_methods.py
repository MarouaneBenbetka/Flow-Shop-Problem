import numpy as np

def calculate_makespan(processing_times, sequence):
    n_jobs = len(sequence)
    n_machines = len(processing_times[0])
    end_time = [[0] * (n_machines + 1) for _ in range(n_jobs + 1)]

    for j in range(1, n_jobs + 1):
        for m in range(1, n_machines + 1):
            end_time[j][m] = max(end_time[j][m - 1], end_time[j - 1]
                                 [m]) + processing_times[sequence[j - 1]][m - 1]

    return end_time[n_jobs][n_machines]


# ================ NEH ================
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

# ================ HAM ================
def _ham_sol_1(Pi1: np.ndarray, Pi2: np.ndarray) -> list:
    diff = Pi2 - Pi1
    sol = np.argsort(diff, axis=0).tolist()
    sol.reverse()       # in decreasing order
    return sol

def _ham_sol_2(Pi1: np.ndarray, Pi2: np.ndarray) -> list:

    diff = Pi2 - Pi1

    Pi1_index = [(x, i) for i, x in enumerate(Pi1) if diff[i] >= 0]
    Pi2_index = [(x, i) for i, x in enumerate(Pi2) if diff[i] < 0]

    Pi1_sorted = sorted(Pi1_index, key=lambda x: x[0])
    Pi2_sorted = sorted(Pi2_index, key=lambda x: x[0], reverse=True)

    Pi1_list = [i for _, i in Pi1_sorted]
    Pi2_list = [i for _, i in Pi2_sorted]

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
    

# ================ PRSKE ================
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