import math

import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import itertools



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
            incremental_cost[0][i] = incremental_cost[0][i - 1 ] + self.current_instance[0][partial_path[i]]

        # evaluate the rest of machines
        for i in range(1, nb_machines):
            incremental_cost[i, 0] = incremental_cost[i - 1, 0] + self.current_instance[i, partial_path[0]]
            for j in range(1, nb_jobs):
            
            
                incremental_cost[i, j] = self.current_instance[i, partial_path[j]] + \
                max(incremental_cost[i - 1, j],incremental_cost[i, j - 1])

        # print(incremental_cost[nb_machines - 1, nb_jobs - 1])

        return incremental_cost[nb_machines - 1, nb_jobs - 1]
    
    
    def H12(self,partial_path,remaining_jobs):
        
        nb_machines = self.current_instance.shape[0]   
        
        
        LBi = 0  
        
        for i in range(nb_machines):
        
        #    bi = np.min(np.sum(self.current_instance[:i, partial_path], axis=0))
                
                if len(remaining_jobs) > 0:
                    # Ti = np.sum(self.current_instance[i:i+1, list(remaining_jobs)],axis=1)[0]
                    ai = np.min(np.sum(self.current_instance[i+1:, list(remaining_jobs)], axis=0))
                    
                    
                else:
                    
                    return 0
                    ai = 0
                    Ti = 0
            
                #new_lbi = ai + Ti
                new_lbi = ai

                # print(bi, "bi")
                # print(ai, "ai")
                # print(Ti, "Ti")
                # print(new_lbi, "new_lbi")
                
                LBi = max(LBi,new_lbi)  
        
        # print(LBi, "LBi")
        
    
        return LBi
    
    
    def H(self,partial_path,remaining_jobs):
        print('called heuristic on:', partial_path, remaining_jobs)
        # we have already taken the partial path
        # we need to estimate the remaining cost in an optimistic approach
        # which means that we consider the mininum execution time for the bottleneck machine
        # in other words, for each machine, we compute the minimal remaining execution time
        # and take the maximum out of them
        full_path = partial_path + remaining_jobs
        
        nb_jobs = self.current_instance.shape[1]
        nb_machines = self.current_instance.shape[0]      
        
        costs = []
        for machine in range(nb_machines):
            current_machine_cost = 0
            # we will consider doing the following computations
            # for the first machine, we will take only all the remaining jobs into consideration
            # for the second machine, we will consider the remaining jobs + the last job in the partial path
            # for the third machine, we will consider the remaining jobs + the last two jobs in the partial path
            # for the fourth machine, we will consider the remaining jobs + the last three jobs in the partial path
            # and so on
            # starting from a certain machine, we will consider the remaining jobs + all the jobs in the partial path
            # and so on
            # we will take the maximum of all these values
            # and this will be the lower bound
            # write the codef or this logic
            # print("machine paria nebdaw belevel",len(partial_path), max(0, len(partial_path) - machine  ), nb_jobs)
            for i in range(max(0, len(partial_path) ), nb_jobs):
                
                current_machine_cost += self.current_instance[machine, full_path[i]]
            costs.append(current_machine_cost)
            print('got result for machine', machine, current_machine_cost)
        return max(costs) 
            
        
        

    
    
    def LB(self):
        
        '''
        - I have partial sequence, loop through ga3 les machines
        -
        -
        -
        
        '''

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
                    
                    
                    
                    

                # adding the constructed child to the list
                

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



    def lower_bound(self,sequence, remaining_jobs): 
        lower_bound = self.current_instance[1:,sequence].max() + self.current_instance[:,list(remaining_jobs)].sum(axis=1).max()
        return lower_bound
    
    def evaluate_sequence(self,sequence):
        machines = self.current_instance.shape[0]
        machine_times = np.zeros(machines)
        for job in sequence:
            machine_times[0] += self.current_instance[0][job]
            for i in range(1, machines):
                machine_times[i] = max(machine_times[i-1], machine_times[i]) + self.current_instance[i][job]
        return machine_times[-1]
    

    def solve(self, initial_solution):
        # create the root node and append it to the list of active nodes\
        
        self.intial_solution = initial_solution
        initial_bound = self.compute_lower_bound(initial_solution)
        self.bound = initial_bound
        
        
        
        job_count = self.current_instance.shape[1]
        
        root = Node(
            partial_job_order=[],
            bound=initial_bound,
            remaining_jobs=list(range(job_count)),
            level=0,
            child_nodes=[],
            cost=0    
        )
        
        
        for i in root.remaining_jobs:
            child_partial_path = root.partial_job_order + [i]
            child_remaining_jobs = list(set(root.remaining_jobs) - {i})
            
            child_node = Node(
            bound=None,
            child_nodes=None,
            parent=root,
            level=root.level + 1,
            cost=None,
            partial_job_order=child_partial_path,
            remaining_jobs=child_remaining_jobs
            
        )
        
            self.active_nodes.append(child_node)
        
        
        
        while(self.active_nodes):
            
            node = self.active_nodes.pop(0)
            
            
            # check the current nodei s a leaf
            # thel lengrth of the remainging jobs is 0, if the current node is leaf
            if len(node.remaining_jobs)  == 0:
                # compute the real cost
                real_cost = self.compute_lower_bound(node.partial_job_order)
                # if the real cost is better than the cost of the solution already found:
                if real_cost < self.bound:
                    # update the solution with the current solution
                    self.intial_solution = node.partial_job_order
                    # update the lower bound with the cost of the current solution
                    self.bound = real_cost
                # else (the real cost is worse than the cos tof the solution alread found=)
                else:
                    # nothing interesting to do
                    continue
            # else: (the current node is not a leaf)
            else:
                # evaluate the lower bound of the curren node
                print(self.H(node.partial_job_order,node.remaining_jobs))
                print(self.compute_lower_bound(node.partial_job_order))
                estimated_lower_bound  = self.compute_lower_bound(node.partial_job_order) + self.H(node.partial_job_order,node.remaining_jobs)
                # if the lower bound is worse than the cost of the soultion already found
                if estimated_lower_bound > self.bound:
                    # elagage: do not explore the children of the current nod 
                    continue
                    # --> do not create then children of the current nod
                    
                # else if the lower bound of the current node is lower than the actual bound then:
                else:
                    for n in node.remaining_jobs:
                        child_partial_path = node.partial_job_order + [n]
                        child_remaining_jobs = list(set(node.remaining_jobs) - {n})
                        # create and push the children of the current node
                        child_node = Node(
                            bound=None,
                            child_nodes=None,
                            parent=node,
                            level=node.level + 1,
                            cost=None,
                            partial_job_order=child_partial_path,
                            remaining_jobs=child_remaining_jobs
                            
                        )
                        
                        self.active_nodes.insert(0,child_node)

        print("Optimal Solution:", self.intial_solution)
        print("Optimal Cost:", self.bound)
        # self.generate_gantt_chart(self.intial_solution)
        
        
        
        def solve2(self, initial_solution):
        # create the root node and append it to the list of active nodes\
        
            self.intial_solution = initial_solution
            initial_bound = self.compute_lower_bound(initial_solution)
            self.bound = initial_bound
            
            
            
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
                
                node = self.active_nodes.pop(0)
                
            #     children = self.generate_children(current_node)
                
            #     current_node.childre_nodes = children
            #     self.active_nodes.extend(children)
            

                remaining_jobs_ordered = node.remaining_jobs[::-1]
                for n in remaining_jobs_ordered:
                    self.active_nodes.insert(0,n)

                    
                    child_job_order = node.partial_job_order + [n]
                    # current_bound = self.compute_lower_bound(child_job_order, self.current_instance)
                    child_remaining_jobs = set(node.remaining_jobs ) - {n}
                    # print("child remainign jobs",child_remaining_jobs)
                    # print("partial job order", child_job_order)
                    
                    child_cost = self.compute_lower_bound(child_job_order)
                    # print(self.bound)
                    # print(child_cost, "child cost")

                    if (len(child_remaining_jobs) == 0):
                        print(self.bound, "updated cost")

                        if (child_cost < self.bound):

                            self.intial_solution = child_job_order
                            self.bound = child_cost
                            
                            # print(self.intial_solution, "child job order")
                            print(self.bound, "updated cost")

                    else:
                        if (child_cost < self.bound):

                            # self.intial_solution = child_job_order
                            # self.bound = child_cost
                            
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
                            self.active_nodes.append(current_child)

                        else:
                            print("DKSFJALSKDJF;AKLJL;KSADJKLFJASD;KLFJDSALKJ DSAFKJFLAK;SDJF")
                        
                        #     self.intial_solution = child_job_order
                    #     self.bound = child_cost
                    
                    # children.append(current_child)

        print("Optimal Solution:", self.intial_solution)
        print("Optimal Cost:", self.bound)

        
        
        '''
        
        dfs : 

            on a une 
        
        
        '''