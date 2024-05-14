import numpy as np
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import calculate_makespan
import random
from utils.heuristics import *


def selection_AG(population, processing_times, n_selected, strategie):
    """
    Sélectionne une sous-population en fonction de la stratégie spécifiée.

    Args:
        population (list): La population actuelle de chromosomes (solutions).
        processing_times (array): Matrice des temps de traitement par machine et par job.
        n_selected (int): Nombre d'individus à sélectionner.
        strategie (str): Stratégie de sélection ('roulette', 'Elitism', 'rank', 'tournament').

    Returns:
        list: La sous-population sélectionnée selon la stratégie.
    """

    # Sélection par roulette:
    if strategie == "roulette":
        # Calcul de la fitness pour chaque individu dans la population
        fitness = [calculate_makespan(processing_times, sequence) for sequence in population]
        # Somme des valeurs de fitness pour normalisation
        fitness_sum = sum(fitness)
        # Calcul des probabilités de sélection proportionnelles à la fitness
        selection_probs = [fitness[i] / fitness_sum for i in range(len(population))]
        # Cumul des probabilités pour faciliter la sélection aléatoire
        cum_probs = [sum(selection_probs[:i+1]) for i in range(len(population))]
        selected = []

        # Sélection de n individus en fonction des probabilités cumulées
        for i in range(n_selected):
            while True:
                rand = random.random()
                for j, cum_prob in enumerate(cum_probs):
                    if rand < cum_prob:
                        break
                # Assure que chaque individu sélectionné est unique
                if population[j] not in selected:
                    selected.append(population[j])
                    break

    # Sélection par élitisme:
    elif strategie == "Elitism":
        # Évaluation de la fitness pour toute la population
        fitness = [calculate_makespan(processing_times, sequence) for sequence in population]
        # Tri de la population basé sur la fitness
        sorted_population = [x for x, _ in sorted(zip(population, fitness), key=lambda pair: pair[1])]
        # Sélection des meilleurs individus
        selected = sorted_population[:n_selected]

    # Sélection par classement:
    elif strategie == "rank":
        # Évaluation de la fitness pour toute la population
        fitness = [calculate_makespan(processing_times, sequence) for sequence in population]
        # Tri de la population par fitness
        sorted_population = sorted(population, key=lambda x: fitness[population.index(x)])
        # Calcul des poids pour la sélection aléatoire
        fitness_sum = sum(i + 1 for i in range(len(sorted_population)))
        selection_probs = [(len(sorted_population) - i) / fitness_sum for i in range(len(sorted_population))]
        selected = []

        # Sélection de n individus en fonction des poids
        for i in range(n_selected):
            selected_index = random.choices(range(len(sorted_population)), weights=selection_probs)[0]
            selected.append(sorted_population[selected_index])
            # Suppression pour éviter duplication dans les choix
            sorted_population.pop(selected_index)
            selection_probs.pop(selected_index)

    # Sélection par tournoi:
    elif strategie == "tournament":
        k = 2  # Taille du tournoi
        selected = []

        # Répéter jusqu'à ce que n individus soient sélectionnés
        for i in range(n_selected):
            while True:
                # Choix aléatoire de k individus
                tournament = random.sample(population, k)
                # Filtrer les individus déjà sélectionnés
                tournament = [seq for seq in tournament if seq not in selected]
                if tournament:
                    break
            # Évaluation de la fitness des individus du tournoi
            fitness = [calculate_makespan(processing_times, sequence) for sequence in tournament]
            # Sélection de l'individu avec la meilleure fitness
            selected.append(tournament[fitness.index(min(fitness))])

    return selected

def remove_duplicates_AG(enfant, other_enfant, points):
    # Calcul de la taille de la liste des jobs, moins un pour obtenir l'indice maximum utilisable.
    jobs = len(enfant) - 1

    # Détermine si les points de crossover fournis sont multiples (typiquement pour un crossover à deux points).
    check_points = len(points) > 1

    # Boucle continue jusqu'à ce que tous les duplicats soient résolus.
    while True:
        # Crée un ensemble de jobs qui apparaissent plus d'une fois dans l'offspring.
        duplicates = set([job for job in enfant if enfant.count(job) > 1])

        # Si aucun duplicat n'est trouvé, sortir de la boucle.
        if not duplicates:
            break

        # Traiter chaque job dupliqué trouvé.
        for job in duplicates:
            # Trouve toutes les positions de ce job dans l'offspring.
            pos = [i for i, x in enumerate(enfant) if x == job]

            # Détermine quelle position du job dupliqué doit être corrigée.
            # Si des points de crossover sont spécifiés et que la première occurrence du job dupliqué est hors de ces points,
            # ou si aucun point n'est spécifié et que la première occurrence est avant le premier point,
            # alors 'dup' est la première position, sinon c'est la seconde.
            if (check_points and ((pos[0] < points[0]) or (pos[0] >= points[1])) ) or ( (pos[0] < points[0]) and not check_points):
                dup = pos[0]
                index = pos[1]
            else:
                dup = pos[1]
                index = pos[0]

            # Remplace le job dupliqué à la position 'dup' par un job de 'other_offspring' à la position 'index'.
            # Cette substitution est faite pour corriger la duplication tout en essayant de préserver la structure génétique du parent non-dupliqué.
            enfant[dup] = other_enfant[index]

    # Retourne l'offspring corrigé.
    return enfant

def crossover_AG(parent1, parent2, points):
    # Calcul de l'indice maximal utilisable pour le point de crossover, basé sur la longueur du parent.
    jobs = len(parent1) - 1

    # Si le mode de crossover est à un seul point ("ONE").
    if points == 'ONE':
        # Choix aléatoire d'un point de crossover dans la plage valide.
        point = random.randint(0, jobs)
        # Création de deux enfants en combinant les segments des deux parents autour du point de crossover.
        enfant1 = parent1[:point] + parent2[point:]
        enfant2 = parent2[:point] + parent1[point:]
        # Mise à jour des points pour la suppression des doublons.
        points = [point]

    else:  # Si le mode de crossover est à deux points.
        # Choix aléatoire de deux points de crossover.
        point_1 = random.randint(0, jobs)
        point_2 = random.randint(0, jobs)
        # Assurer que point_1 est inférieur à point_2.
        if point_1 > point_2:
            point_1, point_2 = point_2, point_1
        # Création des enfants en échangeant les segments entre les deux points choisis.
        enfant1 = parent1[:point_1] + parent2[point_1:point_2] + parent1[point_2:]
        enfant2 = parent2[:point_1] + parent1[point_1:point_2] + parent2[point_2:]
        # Mise à jour des points pour la suppression des doublons.
        points = [point_1, point_2]

    # Appel de la fonction pour retirer les doublons, en passant les enfants créés et les points de crossover.
    enfant1 = remove_duplicates_AG(enfant1, enfant2, points)
    enfant2 = remove_duplicates_AG(enfant2, enfant1, points)

    # Retourne les deux enfants après élimination des duplications.
    return enfant1, enfant2

def mutation_AG(sequence, mutation_rate):
    # Calcul de la longueur de la séquence, correspondant au nombre de jobs ou tâches
    num_jobs = len(sequence)
    
    # Parcours de chaque élément de la séquence pour potentiellement le muter
    for i in range(num_jobs):
        # Génération d'un nombre aléatoire entre 0 et 1
        r = random.random()
        
        # Vérification si ce nombre est inférieur au taux de mutation
        if r < mutation_rate:
            # Création d'une liste des indices de tous les jobs sauf celui à l'index i pour éviter de muter un job avec lui-même
            available_jobs = [j for j in range(num_jobs) if j != sequence[i]]
            
            # Sélection aléatoire d'un nouvel index de job parmi les disponibles
            newjob = random.sample(available_jobs, 1)[0]
            
            # Echange des positions dans la séquence pour introduire la mutation
            sequence[sequence.index(newjob)], sequence[i] = sequence[i], newjob
    
    # Retour de la séquence mutée après potentiellement plusieurs mutations
    return sequence

def genetic_algorithm(processing_times, init_pop, pop_size, select_pop_size, selection_method, crossover, mutation_probability, num_iterations):

    # Init population generation
    population = init_pop
    best_seq = selection_AG(population, processing_times, 1, "Elitism")[0]
    best_cost = calculate_makespan(processing_times, best_seq)
    for i in range(num_iterations):
        # Selection
        s = int(select_pop_size * pop_size) # number of selected individus to be parents (%)
        parents = selection_AG(population, processing_times, s, selection_method)
        # Crossover
        new_generation = []
        for _ in range(0, pop_size, 2):
            parent1 = random.choice(parents)
            parent2 = random.choice([p for p in parents if p != parent1])
            child1, child2 = crossover_AG(parent1, parent2, crossover)
            new_generation.append(child1)
            new_generation.append(child2)

        new_generation = new_generation[:pop_size]
        # Mutation
        for i in range(pop_size):
            if random.uniform(0, 1) < mutation_probability:
                new_generation[i] = mutation_AG(new_generation[i], mutation_probability)
        # Replacement
        population = new_generation

        # checking for best seq in current population
        best_seq_pop = selection_AG(population, processing_times, 1, "Elitism")[0]
        best_cost_pop = calculate_makespan(processing_times, best_seq_pop)
        if best_cost_pop < best_cost:
            best_seq = best_seq_pop.copy()
            best_cost = best_cost_pop

    return best_seq


class AG:
    def __init__(self, processing_times, init_pop, pop_size = 30, select_pop_size = 0.5, selection_method = "roulette", crossover = "TWO", mutation_probability = 0.05, num_iterations = 100):
        self.processing_times = processing_times
        self.init_pop = init_pop
        self.pop_size = pop_size
        self.select_pop_size = select_pop_size
        self.selection_method = selection_method
        self.crossover = crossover
        self.mutation_probability = mutation_probability
        self.num_iterations = num_iterations

    # def selection_AG(self, n_selected):
    #     return selection_AG(self.init_pop, self.processing_times, n_selected, self.selection_method)
    
    # def remove_duplicates_AG(self, enfant, other_enfant, points):
    #     return remove_duplicates_AG(enfant, other_enfant, points)

    # def crossover_AG(self, parent1, parent2, points):
    #     return crossover_AG(parent1, parent2, points)

    # def mutation_AG(self, sequence, mutation_rate):
    #     return mutation_AG(sequence, mutation_rate)

    def run(self):
        # Init population generation
        population = self.init_pop
        best_seq = selection_AG(population, self.processing_times, 1, "Elitism")[0]
        best_cost = calculate_makespan(self.processing_times, best_seq)
        for i in range(self.num_iterations):
            # Selection
            s = int(self.select_pop_size * self.pop_size) # number of selected individus to be parents (%)
            parents = selection_AG(population, self.processing_times, s, self.selection_method)
            # Crossover
            new_generation = []
            for _ in range(0, self.pop_size, 2):
                parent1 = random.choice(parents)
                parent2 = random.choice([p for p in parents if p != parent1])
                child1, child2 = crossover_AG(parent1, parent2, self.crossover)
                new_generation.append(child1)
                new_generation.append(child2)

            new_generation = new_generation[:self.pop_size]
            # Mutation
            for i in range(self.pop_size):
                if random.uniform(0, 1) < self.mutation_probability:
                    new_generation[i] = mutation_AG(new_generation[i], self.mutation_probability)
            # Replacement
            population = new_generation

            # checking for best seq in current population
            best_seq_pop = selection_AG(population, self.processing_times, 1, "Elitism")[0]
            best_cost_pop = calculate_makespan(self.processing_times, best_seq_pop)
            if best_cost_pop < best_cost:
                best_seq = best_seq_pop.copy()
                best_cost = best_cost_pop

        return best_seq

class ArtificialBeeColony:
    def __init__(self, ProcessingTimes, MaxIt=200, nPop=100, nOnlooker=None, L=None):
        self.ProcessingTimes = ProcessingTimes
        self.MaxIt = MaxIt
        self.nPop = nPop
        self.nOnlooker = nOnlooker if nOnlooker is not None else nPop
        self.L = L if L is not None else round(0.6 * self.ProcessingTimes.shape[0] * nPop)
    
    def update_solution(self, A, B, psi):
        a_solution, a_cost = A["Solution"], A["Cost"]
        b_solution, b_cost = B["Solution"], B["Cost"]
        intermediate_solution = a_solution.copy()
        BestSol_l = {'Solution': [], "Cost": float('inf')}
        #print('==========================================================================================')
        #print('A: ', a_solution, a_cost)
        #print('B: ', b_solution, b_cost)
        i = 0
        while i < len(b_solution)-2:
            if psi < 0.5:
                i_i_index = intermediate_solution.index(b_solution[i])
                b_i_index = i
            else:
                b_i_index = b_solution.index(i)
                i_i_index = intermediate_solution.index(i)
            if b_i_index != i_i_index:
                intermediate_solution[b_i_index], intermediate_solution[i_i_index] = intermediate_solution[i_i_index], intermediate_solution[b_i_index]
                intermediate_cost = calculate_makespan(self.ProcessingTimes, intermediate_solution)
                if intermediate_cost < BestSol_l["Cost"]:
                    BestSol_l["Solution"] = intermediate_solution.copy()
                    BestSol_l["Cost"] = intermediate_cost
            i+=1
                
        return BestSol_l
    
    def run(self):

        start_time = time.time()
        
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
            F = np.exp(-np.array([x['Cost'] for x in pop]) / np.mean([x['Cost'] for x in pop]))
            P = F / np.sum(F)
            
            # Onlooker Bees
            for m in range(self.nOnlooker):
                #i = self.roulette_wheel_selection(P)
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
                    pop[i]['Cost'] = calculate_makespan(self.ProcessingTimes, pop[i]['Solution'])
                    C[i] = 0
            
            # Update Best Solution Ever Found
            BestSol = min(pop + [BestSol], key=lambda x: x['Cost'])
            # Store Best Cost Ever Found
            BestCost[it] = BestSol['Cost']
            
            # Display Iteration Information
            # print(f"Iteration {it + 1}: Best Cost = {BestCost[it]}")
        print("Elapsed time:", time.time()-start_time, "seconds")
        return BestSol['Solution']
    
    def initialize_population(self):
        pop = []
        for _ in range(self.nPop):
            bee = {'Solution': self.initialize_a_solution()}
            bee['Cost'] = calculate_makespan(self.ProcessingTimes, bee['Solution'])
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
        def generate_heuristic_solution(processing_times ,name = "NEH" ):
            return algorithms[name](processing_times)[0]
        
        solution = generate_heuristic_solution(self.ProcessingTimes, name=random.choice(list(algorithms.keys())))
        return list(solution)
    
    def roulette_wheel_selection(self, P):
        r = np.random.rand()
        C = np.cumsum(P)
        for i in range(len(C)):
            if r <= C[i]:
                return i