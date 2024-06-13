import numpy as np
import gym
from gym import spaces
import random


class FSSPEnvironment(gym.Env):
    def __init__(self, jobs, machines):
        super(FSSPEnvironment, self).__init__()
        self.jobs = jobs
        self.machines = machines
        self.n_jobs = len(jobs)
        self.n_machines = len(machines)

        # Example: 3 metaheuristics x 10 parameter configurations
        self.action_space = spaces.Discrete(30)
        self.observation_space = spaces.Box(low=0, high=1, shape=(
            self.n_jobs, self.n_machines), dtype=np.float32)

        self.state = np.zeros((self.n_jobs, self.n_machines))

    def reset(self):
        self.state = np.zeros((self.n_jobs, self.n_machines))
        return self.state

    def step(self, action):
        metaheuristic, hyperparameters = self.decode_action(action)
        solution = self.apply_metaheuristic(metaheuristic, hyperparameters)

        makespan = self.calculate_makespan(self.jobs, solution)
        reward = -makespan

        self.state = self.update_state(solution)

        done = True

        return self.state, reward, done, {"metaheuristic": metaheuristic, "hyperparameters": hyperparameters, "makespan": makespan}

    def decode_action(self, action):
        metaheuristic = action // 10
        hyperparameter_set = action % 10
        # Placeholder, should be based on actual parameter ranges
        hyperparameters = {"set": hyperparameter_set}
        return metaheuristic, hyperparameters

    def apply_metaheuristic(self, metaheuristic, hyperparameters):
        if metaheuristic == 0:
            return self.genetic_algorithm(hyperparameters)
        elif metaheuristic == 1:
            return self.simulated_annealing(hyperparameters)
        elif metaheuristic == 2:
            return self.tabu_search(hyperparameters)

    def update_state(self, solution):
        return solution

    def calculate_makespan(self, processing_times, sequence):
        n_jobs = len(sequence)
        n_machines = len(processing_times[0])
        end_time = [[0] * (n_machines + 1) for _ in range(n_jobs + 1)]

        for j in range(1, n_jobs + 1):
            for m in range(1, n_machines + 1):
                end_time[j][m] = max(
                    end_time[j][m - 1], end_time[j - 1][m]) + processing_times[sequence[j - 1]][m - 1]

        return end_time[n_jobs][n_machines]

    # Genetic Algorithm (GA) Implementation
    def genetic_algorithm(self, hyperparameters):
        population_size = hyperparameters.get("population_size", 10)
        mutation_rate = hyperparameters.get("mutation_rate", 0.1)
        generations = hyperparameters.get("generations", 50)
        # TODO : taux de croisement ,

        # Initialize population
        population = [np.random.permutation(
            self.n_jobs) for _ in range(population_size)]
        best_sequence = None
        best_makespan = float('inf')

        for _ in range(generations):
            # Evaluate population
            fitness = [self.calculate_makespan(
                self.jobs, sequence) for sequence in population]
            best_index = np.argmin(fitness)
            if fitness[best_index] < best_makespan:
                best_makespan = fitness[best_index]
                best_sequence = population[best_index]

            # Selection (tournament selection)
            selected = []
            for _ in range(population_size):
                i, j = random.sample(range(population_size), 2)
                selected.append(
                    population[i] if fitness[i] < fitness[j] else population[j])

            # Crossover (order crossover)
            next_population = []
            for i in range(0, population_size, 2):
                if i + 1 < population_size:
                    parent1, parent2 = selected[i], selected[i + 1]
                    cut = random.randint(1, self.n_jobs - 1)
                    child1 = np.concatenate(
                        (parent1[:cut], [job for job in parent2 if job not in parent1[:cut]]))
                    child2 = np.concatenate(
                        (parent2[:cut], [job for job in parent1 if job not in parent2[:cut]]))
                    next_population.extend([child1, child2])
                else:
                    next_population.append(selected[i])

            # Mutation
            for individual in next_population:
                if random.random() < mutation_rate:
                    i, j = random.sample(range(self.n_jobs), 2)
                    individual[i], individual[j] = individual[j], individual[i]

            population = next_population

        return best_sequence

    # Simulated Annealing (SA) Implementation
    def simulated_annealing(self, hyperparameters):
        initial_temperature = hyperparameters.get("initial_temperature", 100)
        cooling_rate = hyperparameters.get("cooling_rate", 0.95)
        max_iterations = hyperparameters.get("max_iterations", 1000)

        current_sequence = np.random.permutation(self.n_jobs)
        current_makespan = self.calculate_makespan(self.jobs, current_sequence)
        best_sequence = current_sequence
        best_makespan = current_makespan
        temperature = initial_temperature

        for _ in range(max_iterations):
            if temperature < 1e-3:
                break

            # Generate neighbor
            i, j = random.sample(range(self.n_jobs), 2)
            neighbor_sequence = current_sequence.copy()
            neighbor_sequence[i], neighbor_sequence[j] = neighbor_sequence[j], neighbor_sequence[i]
            neighbor_makespan = self.calculate_makespan(
                self.jobs, neighbor_sequence)

            # Acceptance probability
            if neighbor_makespan < current_makespan or random.random() < np.exp((current_makespan - neighbor_makespan) / temperature):
                current_sequence = neighbor_sequence
                current_makespan = neighbor_makespan

            # Update best solution
            if current_makespan < best_makespan:
                best_sequence = current_sequence
                best_makespan = current_makespan

            temperature *= cooling_rate

        return best_sequence

    # Tabu Search (TS) Implementation
    def tabu_search(self, hyperparameters):
        tabu_list_size = hyperparameters.get("tabu_list_size", 10)
        max_iterations = hyperparameters.get("max_iterations", 100)

        current_sequence = np.random.permutation(self.n_jobs)
        current_makespan = self.calculate_makespan(self.jobs, current_sequence)
        best_sequence = current_sequence
        best_makespan = current_makespan

        tabu_list = []

        for _ in range(max_iterations):
            neighborhood = []
            for i in range(self.n_jobs):
                for j in range(i + 1, self.n_jobs):
                    neighbor_sequence = current_sequence.copy()
                    neighbor_sequence[i], neighbor_sequence[j] = neighbor_sequence[j], neighbor_sequence[i]
                    neighborhood.append((neighbor_sequence, (i, j)))

            neighborhood = [(seq, swap)
                            for seq, swap in neighborhood if swap not in tabu_list]
            if not neighborhood:
                break

            neighborhood_fitness = [self.calculate_makespan(
                self.jobs, seq) for seq, _ in neighborhood]
            best_neighbor_index = np.argmin(neighborhood_fitness)
            best_neighbor_sequence, best_neighbor_swap = neighborhood[best_neighbor_index]
            best_neighbor_makespan = neighborhood_fitness[best_neighbor_index]

            if best_neighbor_makespan < best_makespan:
                best_sequence = best_neighbor_sequence
                best_makespan = best_neighbor_makespan

            current_sequence = best_neighbor_sequence
            current_makespan = best_neighbor_makespan
            tabu_list.append(best_neighbor_swap)

            if len(tabu_list) > tabu_list_size:
                tabu_list.pop(0)

        return best_sequence
