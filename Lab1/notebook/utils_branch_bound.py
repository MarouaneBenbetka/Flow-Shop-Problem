import math

import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import itertools
import time


'''
Reading the data from the benchmark file
'''

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


# the node represents a partial solution in our case
class Node(object):
    def __init__(self, partial_job_order, bound, remaining_jobs, level, child_nodes, cost, parent=None):
        self.partial_job_order = partial_job_order
        self.bound = bound
        self.level = level
        self.child_nodes = []
        self.remaining_jobs = remaining_jobs
        self.cost = cost
        self.parent = parent
    # def evaluate_solution(self, solution):
    #     # use the processing times matrix to compute the value of the solution
    #     pass

    def __str__(self):
        # present the node in a readable format
        return ""


def johnson_method(processing_times):
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

    return m1 + m2

class FlowShopBranchBoundSolver(object):

    def __init__(self):
        self.active_nodes = []
        self.bound = math.inf
        self.current_instance = None
        self.order_heuristic = None
        self.evaluation = None
        self.intial_solution = None

    def consider_instance(self, instance):
        self.current_instance = instance

    def consider_heuristic(self, heuristic):
        self.order_heuristic = heuristic

    def consider_evaluation(self, evaluation):
        self.evaluation = evaluation

    def calculate_bi(self, machine_i):
        if machine_i > 0:
            total_times = np.sum(self.current_instance[:machine_i, :], axis=0)
            return np.min(total_times)
        else:
            return 0

    def calculate_ai(self, machine_i):
        if machine_i < self.current_instance.shape[0]-1:
            total_times = np.sum(self.current_instance[machine_i+1:, :], axis=0)
            return np.min(total_times)
        else:
            return 0

    def calculate_ti(self, machine_i):
        return np.sum(self.current_instance[machine_i, :], axis=0)

    def calculate_LB(self):
        LB = 0
        machine = 0
        for i in range(self.current_instance.shape[0]):
            bi = self.calculate_bi(i)
            ai = self.calculate_ai(i)
            ti = self.calculate_ti(i)
            if ai + ti + bi > LB:
                LB = ai + ti + bi
                machine = i

        return LB, machine


    def calculate_makespan(self, partial_path):
        
        # print("called function with partial path : ", partial_path)
        # compute the value of the lower bound from the instance matrix
        # there is a ready formula that we can use directly

        nb_jobs = len(partial_path)
        nb_machines = self.current_instance.shape[0]        
        
        incremental_cost = np.zeros((nb_machines, nb_jobs))
        



        # evaluate the first machines

        incremental_cost[0, 0] = self.current_instance[0][partial_path[0] ]

        for i in range(1, nb_jobs):
            incremental_cost[0][i] = incremental_cost[0][i - 1 ] + self.current_instance[0][partial_path[i]]

        # evaluate the rest of machines
        for i in range(1, nb_machines):
            incremental_cost[i, 0] = incremental_cost[i - 1, 0] + self.current_instance[i, partial_path[0]]
            for j in range(1, nb_jobs):
            
            
                incremental_cost[i, j] = self.current_instance[i, partial_path[j]] + \
                max(incremental_cost[i - 1, j],incremental_cost[i, j - 1])

        # print(incremental_cost[nb_machines - 1, nb_jobs - 1])

        return incremental_cost[nb_machines - 1, nb_jobs - 1]

    def compute_lower_bound_2(self, partial_path, sums):
        
        # print("called function with partial path : ", partial_path)
        # compute the value of the lower bound from the instance matrix
        # there is a ready formula that we can use directly

        nb_jobs = len(partial_path)
        nb_machines = self.current_instance.shape[0]        
        
        incremental_cost = np.zeros((nb_machines, nb_jobs))
        
        LB = []  

        if nb_jobs > 1:

            # Delete all jobs from the current path to have only the rest
            right = np.delete(self.current_instance, [job for job in partial_path], axis=1)

            # =======================================================
            # evaluate the first machine
            incremental_cost[0, 0] = self.current_instance[0][partial_path[0]]

            for i in range(1, nb_jobs):
                incremental_cost[0][i] = incremental_cost[0][i - 1 ] + self.current_instance[0][partial_path[i]]

            # evaluate the rest of machines
            for i in range(1, nb_machines):
                incremental_cost[i, 0] = incremental_cost[i - 1, 0] + self.current_instance[i, partial_path[0]]
                for j in range(1, nb_jobs):
                
                
                    incremental_cost[i, j] = self.current_instance[i, partial_path[j]] + \
                    max(incremental_cost[i - 1, j],incremental_cost[i, j - 1])

            # =======================================================

            # For the first machine
            right_side = np.sum(right[1:, :], axis=0)
            LB.append(np.min(right_side) + sums[0]) 

            # For the rest of the machines  
            for i in range(1 ,nb_machines):
                cj = incremental_cost[i, -1]

                # Sums the resting jobs on the current machine
                sum_job_on_machine = np.sum(right[i, :])

                # Sum the mins of all resting machines for resting jobs
                rest_sum = 0
                if i < nb_machines-1:
                    rest_sum = np.sum(np.min(right[i+1:, :], axis=1))

                LB.append(cj + sum_job_on_machine + rest_sum)


        
        else: # Only one job
            job = partial_path[0]

            # delete the column of the starting job
            right = np.delete(self.current_instance, job, axis=1)

            for i in range(nb_machines):
                # Calculate the past knowing that we started with job so we don't consider it
                left_side = np.sum(self.current_instance[:i, job], axis=0)

                # Calculate the rest without considering the starting job
                right_side = np.sum(right[i+1:, :], axis=0)
                LB.append(np.min(left_side) + np.min(right_side) + sums[i])

        return max(LB)



    def compute_upper_bound(self, partial_path):
        # use the processing times matrix to compute the upper bound value
        # there is a ready formula that we can use directly
        return 0

    def johnson_method(self, instance):
        # machines, jobs = self.current_instance.shape
        # copy_instance = self.current_instance.copy().T
        # maximum = self.current_instance.max() + 1
        print('instance shape:', instance.shape)
        # machines, jobs = instance.shape
        jobs, machines = instance.shape
        copy_instance = instance.copy().T
        maximum = instance.max() + 1

        m1 = []
        m2 = []

        if machines != 2:
            raise Exception("Johson method only works with two machines")

        for i in range(jobs):
            minimum = copy_instance.min()
            position = np.where(copy_instance == minimum)

            if position[1][0] == 0:
                m1.append(position[0][0])
            else:
                m2.insert(0, position[0][0])

            copy_instance[position[0][0]] = maximum
        return m1+m2

    def generate_children(self, node):
        children = []
        for n in node.remaining_jobs:
            child_job_order = node.partial_job_order + [n]
            # current_bound = self.compute_lower_bound(child_job_order, self.current_instance)
            child_remaining_jobs = set(node.remaining_jobs ) - {n}
            # print("child remainign jobs",child_remaining_jobs)
            # print("partial job order", child_job_order)
            
            child_cost = self.compute_lower_bound(child_job_order)

            if (len(child_remaining_jobs) == 0):
                if (child_cost < self.bound):

                    self.intial_solution = child_job_order
                    self.bound = child_cost
            else:
                if (child_cost < self.bound):
                    current_child = Node(
                        # partial_job_order, bound, remaining_jobs, level, child_nodes, cost, parent=None
                        partial_job_order=child_job_order,
                        bound=None,
                        remaining_jobs=child_remaining_jobs,
                        level=node.level + 1,
                        child_nodes=[],
                        cost=child_cost,
                        parent=node
                    )
                    
                    
                    

                # adding the constructed child to the list
                children.append(current_child)

        return children
    
    
    def all_permutations(self, iterable):
        permutations = list(itertools.permutations(iterable))
        permutations_as_lists = [list(p) for p in permutations]
        return permutations_as_lists

    def generate_gantt_chart(self, solution):
        plt.figure(figsize=(20, 12))
        df = pd.DataFrame(columns=['Machine', 'Job', 'Start', 'Finish'])

        machines, jobs = self.current_instance.shape
        machine_times = np.zeros((machines, jobs))
        start_time_m = np.zeros(machines)
        for job in solution:

            for machine_index in range(machines):
                start_time = start_time_m[machine_index]
                if machine_index > 0:
                    start_time = max(start_time, start_time_m[machine_index-1])
                end_time = start_time + self.current_instance[machine_index, job]
                start_time_m[machine_index] = end_time

                df = pd.concat([df, pd.DataFrame({'Machine': f'Machine {machine_index + 1}',
                                   'Job': f'Job {job + 1}',
                                   'Start': start_time,
                                   'Finish': end_time}, index=[0])], ignore_index=True)

                machine_times[machine_index, job] = end_time

        colors = plt.cm.tab10.colors
        for i, machine_index in enumerate(range(machines)):
            machine_df = df[df['Machine'] == f'Machine {machine_index + 1}']
            plt.broken_barh([(start, end - start) for start, end in zip(machine_df['Start'], machine_df['Finish'])],
                             (i * 10, 9), facecolors=[colors[j % 10] for j in range(jobs)], edgecolor='black')

        plt.xlabel('Time')
        plt.yticks([i * 10 + 4.5 for i in range(machines)], [f'Machine {i + 1}' for i in range(machines)])
        plt.show()



    def solve(self, initial_solution):
        # create the root node and append it to the list of active nodes\
        
        self.intial_solution = initial_solution
        initial_bound = self.compute_lower_bound(initial_solution)
        
        
        
        job_count = self.current_instance.shape[1]
        
        root = Node(
            partial_job_order=[],
            bound=initial_bound,
            remaining_jobs=list(range(job_count)),
            level=0,
            child_nodes=[],
            cost=0    
        )
        self.active_nodes.append(root)
        while(self.active_nodes):
            
            current_node = self.active_nodes.pop(0)
            
            children = self.generate_children(current_node)
            
            current_node.childre_nodes = children
            self.active_nodes.extend(children)
        print("Optimal Solution:", self.intial_solution)
        print("Optimal Cost:", self.bound)
        # self.generate_gantt_chart(self.intial_solution)


    def branch_and_bound_old(self, initial_solution, upper_bound):
        machines, jobs = self.current_instance.shape
        start = time.time()
        # initial_bound, _ = self.calculate_LB()
        initial_bound = self.compute_lower_bound(initial_solution)
        print("Initial bound: ", initial_bound)
        root = Node(
            partial_job_order=[],
            bound=initial_bound,
            remaining_jobs=list(range(jobs)),
            level=0,
            child_nodes=[],
            cost=0    
        )
        best_solution = initial_solution.copy()
        best_cost = initial_bound
        self.active_nodes.append(root)
        i = 0
        while self.active_nodes:
            node = self.active_nodes.pop()

            for job in node.remaining_jobs:
                child_jobs = node.partial_job_order + [job]
                child_remaining_jobs = node.remaining_jobs.copy()
                child_remaining_jobs.remove(job)
                child_lower_bound = self.compute_lower_bound(child_jobs)
                if i % 100000 == 0:
                    print("Partial path: ", child_jobs)
                    print("Child_lower_bound: ", child_lower_bound)
                if not child_remaining_jobs:
                    print("Reached a child")
                    if child_lower_bound < best_cost:
                        print("Good child with lower bound: ", child_lower_bound)
                        best_solution = child_jobs
                        best_cost = child_lower_bound
                        continue
                # If the child node is not a leaf then calculate its lower bound and compare it with current `best_cost`
                if child_lower_bound < best_cost:
                    if child_lower_bound < upper_bound:
                        if i % 100000 == 0:
                            print("Added a node with lower bound: ", child_lower_bound)
                        # child_node = Node(child_jobs, bound=child_lower_bound, child_remaining_jobs, parent=node)
                        child_node = Node(
                                    partial_job_order=child_jobs,
                                    bound=child_lower_bound,
                                    remaining_jobs=child_remaining_jobs,
                                    level=i+1,
                                    child_nodes=[],
                                    cost=0    
                                )
                        self.active_nodes.append(child_node)
            i += 1

        print("End in: ", time.time()- start)
        return best_solution, best_cost, i

    def branch_and_bound(self):
        machines, jobs = self.current_instance.shape
        start = time.time()

        # Calculate initial upper bound
        upper_bound = self.current_instance.sum()

        # Calculate total lower bound
        sums = np.sum(self.current_instance, axis=1)
        LB_total = np.max(sums)

        # Calculate lower bounds with respect to machines
        LB = [LB_total]
        for i in range(machines):
            left_side = 0
            if i > 0:
                left_side = np.sum(self.current_instance[:i, :], axis=0)
                left_side = np.min(left_side)
            right_side = 0
            if i < machines:
                right_side = np.sum(self.current_instance[i+1:, :], axis=0)
                right_side = np.min(right_side)

            LB.append(np.min(left_side) + right_side + sums[i])
                
        
        # Take the max of all LBs
        initial_bound = max(LB)        
        
        root = Node(
            partial_job_order=[],
            bound=initial_bound,
            remaining_jobs=list(range(jobs)),
            level=0,
            child_nodes=[],
            cost=0    
        )

        self.active_nodes.append(root)
        i = 0
        while self.active_nodes:

            node = min(self.active_nodes, key=lambda x: x.bound)
            self.active_nodes.remove(node)

            if node.bound <= upper_bound:

                if len(node.remaining_jobs) > 2:

                    for job in node.remaining_jobs:
                        child_jobs = node.partial_job_order + [job]
                        child_remaining_jobs = node.remaining_jobs.copy()
                        child_remaining_jobs.remove(job)
                        child_lower_bound = self.compute_lower_bound(child_jobs, sums)

                        # If the child node is not a leaf then calculate its lower bound and compare it with current best upper bound
                        if child_lower_bound < upper_bound:
                            child_node = Node(
                                        partial_job_order=child_jobs,
                                        bound=child_lower_bound,
                                        remaining_jobs=child_remaining_jobs,
                                        level=i+1,
                                        child_nodes=[],
                                        cost=0    
                                    )
                            self.active_nodes.append(child_node)

                else:
                    [job1, job2] = node.remaining_jobs
                    child_jobs = node.partial_job_order + [job1, job2]
                    makespan = self.calculate_makespan(child_jobs)
                    if makespan <= upper_bound:
                        best_solution = child_jobs
                        upper_bound = makespan

                    child_jobs = node.partial_job_order + [job2, job1]
                    makespan = self.calculate_makespan(child_jobs)
                    if makespan <= upper_bound:
                        best_solution = child_jobs
                        upper_bound = makespan
            i += 1

        print("End in: ", time.time()- start)
        return best_solution, upper_bound, i

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
            LB = compute_lower_bound(processing_times, new_seq)
            if LB < makespan:
                makespan = LB
                initial_sol = new_seq

        sequence = initial_sol

    return initial_sol, makespan



solver = FlowShopBranchBoundSolver()
processing_times = np.array(
    [
        [26, 38, 27, 88, 95, 55, 54, 63, 23, 45, 86, 43, 43, 40, 37, 54, 35, 59, 43, 50],
        [59, 62, 44, 10, 23, 64, 47, 68, 54, 9, 30, 31, 92, 7, 14, 95, 76, 82, 91, 37],
        [78, 90, 64, 49, 47, 20, 61, 93, 36, 47, 70, 54, 87, 13, 40, 34, 55, 13, 11, 5],
        [88, 54, 47, 83, 84, 9, 30, 11, 92, 63, 62, 75, 48, 23, 85, 23, 4, 31, 13, 98],
        [69, 30, 61, 35, 53, 98, 94, 33, 77, 31, 54, 71, 78, 9, 79, 51, 76, 56, 80, 72]
    ]
)

processing_times = np.array(
    [
        [15, 64, 64, 48, 9, 91, 27, 34, 42, 3, 11, 54, 27, 30, 9, 15, 88, 55, 50, 57],
        [28, 4, 43, 93, 1, 81, 77, 69, 52, 28, 28, 77, 42, 53, 46, 49, 15, 43, 65, 41],
        [77, 36, 57, 15, 81, 82, 98, 97, 12, 35, 84, 70, 27, 37, 59, 42, 57, 16, 11, 34],
        [1, 59, 95, 49, 90, 78, 3, 69, 99, 41, 73, 28, 99, 13, 59, 47, 8, 92, 87, 62],
        [45, 73, 59, 63, 54, 98, 39, 75, 33, 8, 86, 41, 41, 22, 43, 34, 80, 16, 37, 94]
    ]
)


# processing_times = np.array(
#     [
#         [5, 8, 7, 3, 13, 6, 55, 54, 63, 55, 54, 63, 55, 54, 63],
#         [6, 7, 2, 5, 45, 12, 64, 47, 68, 64, 47, 68, 64, 47, 68],
#         [4, 8, 7, 9, 6, 42, 20, 61, 93, 20, 61, 93, 20, 61, 93],
#         [1, 2, 3, 4, 13, 9, 9, 30, 11, 9, 30, 11, 9, 30, 11,],
#         [15, 65, 7, 10, 6, 9, 98, 94, 33, 98, 94, 33, 98, 94, 33]
#     ]
# )

solver.consider_instance(processing_times)


initial_solution , upper_bound = NEH(processing_times)

upper_bound = 1359

# print("First upper bound: ", upper_bound)

best_sol, best_cost, i = solver.branch_and_bound()

# makspan = solver.compute_lower_bound(solver.intial_solution)

print("makspan : ",best_cost)
print("Solution : ", best_sol)
print("Number of iteration: ", i)