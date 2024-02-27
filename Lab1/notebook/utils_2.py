import math
import numpy as np


def read_flow_shop_data(file_path, machine_count, job_count):
    instances = []
    with open(file_path) as p:
        lines = p.readlines()
        line_count = len(lines)

        instance_count = line_count // (machine_count + 3)

        for i in range(instance_count):
            # recover the data of each instance
            params_line = lines[i * (machine_count + 3) + 1]
            job_count, machine_count, initial_seed, upper_bound, lower_bound = list(
                map(lambda x: int(x), params_line.split()))

            # processing_times = [list(map(int, lines[i * (machine_count + 3) + 3])) for line in lines]
            processing_times = np.array([list(map(lambda x: int(x), line.strip().split())) for
                                         line in lines[
                                                 i * (machine_count + 3) + 3:  # start
                                                 i * (machine_count + 3) + 3 + machine_count  # end
                                                 ]
                                         ])

            record = (machine_count, job_count, processing_times)
            instances.append(record)

    return instances


"""
1. in the first place, we have the initial void solution
2. we start filling the solution gradually
3. with each generated partial solution, we evaluate the lower and upper bound 
"""


# Define the NEH heuristic
"""
Example:
    J1  J2  J3  J4
M1  12  6   23  4
M2  8   11  16  9
M3  4   2   18  5

"""

processing_times = np.array([
    [20, 6, 23, 4],
    [8, 3, 16, 9],
    [4, 2, 18, 5]
])


def compute_lower_bound(processing_times, partial_path):
        # compute the value of the lower bound from the instance matrix
        # there is a ready formula that we can use directly

        nb_jobs = len(partial_path)
        nb_machines = processing_times.shape[0]

        incremental_cost = np.zeros((nb_machines, nb_jobs))

        # evaluate the first machines

        incremental_cost[0, 0] = processing_times[0, partial_path[0]]

        for i in range(1, nb_jobs):
            incremental_cost[0, i] = incremental_cost[0, i - 1] + processing_times[0, partial_path[i]]

        # evaluate the rest of machines
        for i in range(1, nb_machines):
            incremental_cost[i, 0] = incremental_cost[i - 1, 0] + processing_times[i, partial_path[0]]
            for j in range(1, nb_jobs):
                incremental_cost[i, j] = processing_times[i, partial_path[j]] + max(incremental_cost[i - 1, j],
                                                                           incremental_cost[i, j - 1])

        return incremental_cost[nb_machines - 1, nb_jobs - 1]


def order_jobs_decreasing_order(processing_times):
    total_times = np.sum(processing_times, axis=0)
    return np.argsort(total_times, axis=0).tolist()

def insert_job(schedule, job, position):
    new_sequence = schedule[:]
    new_sequence.insert(position, job)
    return new_sequence

def NEH(processing_times):
    ordered_seq = order_jobs_decreasing_order(processing_times)
    J1, J2 = ordered_seq[:2]
    sequence = []
    if compute_lower_bound(processing_times, [J1, J2]) < compute_lower_bound(processing_times, [J2, J1]):
        sequence = [J1, J2] 
    else:
        sequence = [J2, J1] 

    del ordered_seq[:2]
    

    for job in ordered_seq:
        makespan = float("inf")
        initial_sol = []
        for j in range(len(sequence)+1):
            new_seq = insert_job(sequence, job, j)
            print(f"new seq job: {job}, iteration {j}: {new_seq}")
            LB = compute_lower_bound(processing_times, new_seq)
            print(f"LB job{job}, iteration {j}: {LB}")
            if LB < makespan:
                makespan = LB
                initial_sol = new_seq

        sequence = initial_sol

    return initial_sol, makespan



print(NEH(processing_times))