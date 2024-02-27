import math

import pandas as pd
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

    def compute_lower_bound(self, partial_path):
        
        # print("called function with partial path : ", partial_path)
        # compute the value of the lower bound from the instance matrix
        # there is a ready formula that we can use directly

        nb_jobs = len(partial_path)
        nb_machines = self.current_instance.shape[0]        
        
        incremental_cost = np.zeros((nb_machines, nb_jobs))
        



        # evaluate the first machines

        incremental_cost[0, 0] = self.current_instance[0][partial_path[0] ]

        for i in range(1, nb_jobs):
            incremental_cost[0][i] = incremental_cost[0][i - 1 ] + self.current_instance[0][i]

        # evaluate the rest of machines
        for i in range(1, nb_machines):
            incremental_cost[i, 0] = incremental_cost[i - 1, 0] + self.current_instance[i, partial_path[0]]
            for j in range(1, nb_jobs):
            
            
                incremental_cost[i, j] = self.current_instance[i, partial_path[j]] + \
                max(incremental_cost[i - 1, j],incremental_cost[i, j - 1])

        # print(incremental_cost[nb_machines - 1, nb_jobs - 1])

        return incremental_cost[nb_machines - 1, nb_jobs - 1]

    def compute_upper_bound(self, partial_path):
        # use the processing times matrix to compute the upper bound value
        # there is a ready formula that we can use directly
        return 0

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
        