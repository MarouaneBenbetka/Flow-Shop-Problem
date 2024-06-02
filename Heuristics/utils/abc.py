from utils.heurstics import neh_algorithm, ham_heuristic, cds_heuristic, gupta_heuristic, run_palmer, PRSKE, special_heuristic, NRH, chen_heuristic
import numpy as np
import random
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import calculate_makespan


class ArtificialBeeColony:
    def __init__(self, ProcessingTimes, MaxIt=200, nPop=100, nOnlooker=None, L=None):
        self.ProcessingTimes = ProcessingTimes
        self.MaxIt = MaxIt
        self.nPop = nPop
        self.nOnlooker = nOnlooker if nOnlooker is not None else nPop
        self.L = L if L is not None else round(
            0.6 * self.ProcessingTimes.shape[0] * nPop)

    def update_solution(self, A, B, psi):
        a_solution, a_cost = A["Solution"], A["Cost"]
        b_solution, b_cost = B["Solution"], B["Cost"]
        intermediate_solution = a_solution.copy()
        BestSol_l = {'Solution': [], "Cost": float('inf')}
        # print('==========================================================================================')
        # print('A: ', a_solution, a_cost)
        # print('B: ', b_solution, b_cost)
        i = 0
        while i < len(b_solution)-2:
            if psi < 0.5:
                i_i_index = intermediate_solution.index(b_solution[i])
                b_i_index = i
            else:
                b_i_index = b_solution.index(i)
                i_i_index = intermediate_solution.index(i)
            if b_i_index != i_i_index:
                intermediate_solution[b_i_index], intermediate_solution[
                    i_i_index] = intermediate_solution[i_i_index], intermediate_solution[b_i_index]
                intermediate_cost = calculate_makespan(
                    self.ProcessingTimes, intermediate_solution)
                if intermediate_cost < BestSol_l["Cost"]:
                    BestSol_l["Solution"] = intermediate_solution.copy()
                    BestSol_l["Cost"] = intermediate_cost
            i += 1

        return BestSol_l

    def run(self):
        # Initialization
        pop = self.initialize_population()
        print(min(pop, key=lambda x: x['Cost']))
        BestSol = min(pop, key=lambda x: x['Cost'])
        C = np.zeros(self.nPop)
        BestCost = np.zeros(self.MaxIt)

        # ABC Main Loop
        for it in range(self.MaxIt):
            # Recruited Bees
            for i in range(self.nPop):
                K = list(range(i)) + list(range(i+1, self.nPop))
                k = np.random.choice(K)
                psi = np.random.uniform(-1, 1)
                if psi < 0:
                    newbee = self.update_solution(pop[k], pop[i], abs(psi))
                else:
                    newbee = self.update_solution(pop[i], pop[k], abs(psi))

                if newbee['Cost'] < pop[i]['Cost']:
                    pop[i] = newbee
                else:
                    C[i] += 1

            # Calculate Fitness Values and Selection Probabilities
            F = np.exp(-np.array([x['Cost'] for x in pop]
                                 ) / np.mean([x['Cost'] for x in pop]))
            P = F / np.sum(F)

            # Onlooker Bees
            for m in range(self.nOnlooker):
                # i = self.roulette_wheel_selection(P)
                i = np.argmax(P)
                P[i] = -np.inf
                K = list(range(i)) + list(range(i+1, self.nPop))
                k = np.random.choice(K)
                psi = np.random.uniform(-1, 1)
                if psi < 0:
                    newbee = self.update_solution(pop[k], pop[i], abs(psi))
                else:
                    newbee = self.update_solution(pop[i], pop[k], abs(psi))

                if newbee['Cost'] < pop[i]['Cost']:
                    pop[i] = newbee
                else:
                    C[i] += 1

            # Scout Bees
            for i in range(self.nPop):
                if C[i] >= self.L:
                    pop[i] = {'Solution': self.initialize_a_solution()}
                    pop[i]['Cost'] = calculate_makespan(
                        self.ProcessingTimes, pop[i]['Solution'])
                    C[i] = 0

            # Update Best Solution Ever Found
            BestSol = min(pop + [BestSol], key=lambda x: x['Cost'])
            # Store Best Cost Ever Found
            BestCost[it] = BestSol['Cost']

            # Display Iteration Information
            print(f"Iteration {it + 1}: Best Cost = {BestCost[it]}")

        return BestSol['Solution']

    def initialize_population(self):
        pop = []
        for _ in range(self.nPop):
            bee = {'Solution': self.initialize_a_solution()}
            bee['Cost'] = calculate_makespan(
                self.ProcessingTimes, bee['Solution'])
            pop.append(bee)
        return pop

    def initialize_a_solution(self):
        def random_solution(processing_times):
            n_jobs = processing_times.shape[0]
            solution = list(range(n_jobs))
            random.shuffle(solution)
            return solution, 0

        algorithms = {
            "NEH":  neh_algorithm,
            "Ham":  ham_heuristic,
            "CDS":  cds_heuristic,
            "Gupta": gupta_heuristic,
            "Palmer":  run_palmer,
            "PRSKE":  PRSKE,
            "Weighted CDS": special_heuristic,
            "NRH": NRH,
            "Chen": chen_heuristic,
            "random": random_solution
        }

        def generate_heuristic_solution(processing_times, name="NEH"):
            return algorithms[name](processing_times)[0]

        solution = generate_heuristic_solution(
            self.ProcessingTimes, name=random.choice(list(algorithms.keys())))
        return list(solution)

    def roulette_wheel_selection(self, P):
        r = np.random.rand()
        C = np.cumsum(P)
        for i in range(len(C)):
            if r <= C[i]:
                return i
