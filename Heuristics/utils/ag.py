import numpy as np
import random
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import calculate_makespan


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
        fitness = [calculate_makespan(processing_times, sequence)
                   for sequence in population]
        # Somme des valeurs de fitness pour normalisation
        fitness_sum = sum(fitness)
        # Calcul des probabilités de sélection proportionnelles à la fitness
        selection_probs = [fitness[i] /
                           fitness_sum for i in range(len(population))]
        # Cumul des probabilités pour faciliter la sélection aléatoire
        cum_probs = [sum(selection_probs[:i+1])
                     for i in range(len(population))]
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
        fitness = [calculate_makespan(processing_times, sequence)
                   for sequence in population]
        # Tri de la population basé sur la fitness
        sorted_population = [x for x, _ in sorted(
            zip(population, fitness), key=lambda pair: pair[1])]
        # Sélection des meilleurs individus
        selected = sorted_population[:n_selected]

    # Sélection par classement:
    elif strategie == "rank":
        # Évaluation de la fitness pour toute la population
        fitness = [calculate_makespan(processing_times, sequence)
                   for sequence in population]
        # Tri de la population par fitness
        sorted_population = sorted(
            population, key=lambda x: fitness[population.index(x)])
        # Calcul des poids pour la sélection aléatoire
        fitness_sum = sum(i + 1 for i in range(len(sorted_population)))
        selection_probs = [(len(sorted_population) - i) /
                           fitness_sum for i in range(len(sorted_population))]
        selected = []

        # Sélection de n individus en fonction des poids
        for i in range(n_selected):
            selected_index = random.choices(
                range(len(sorted_population)), weights=selection_probs)[0]
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
            fitness = [calculate_makespan(
                processing_times, sequence) for sequence in tournament]
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
            if (check_points and ((pos[0] < points[0]) or (pos[0] >= points[1]))) or ((pos[0] < points[0]) and not check_points):
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
        enfant1 = parent1[:point_1] + \
            parent2[point_1:point_2] + parent1[point_2:]
        enfant2 = parent2[:point_1] + \
            parent1[point_1:point_2] + parent2[point_2:]
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
    start_time = time.time()
    # Init population generation
    population = [np.random.permutation(
        processing_times.shape[0]).tolist() for _ in range(pop_size)]

    best_seq = selection_AG(population, processing_times, 1, "Elitism",)[0]
    best_cost = calculate_makespan(processing_times, best_seq)
    for i in range(num_iterations):
        # Selection
        # number of selected individus to be parents (%)
        s = int(select_pop_size * pop_size)
        parents = selection_AG(
            population, processing_times, s, selection_method)
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
                new_generation[i] = mutation_AG(
                    new_generation[i], mutation_probability)
        # Replacement
        population = new_generation

        # checking for best seq in current population
        best_seq_pop = selection_AG(
            population, processing_times, 1, "Elitism")[0]
        best_cost_pop = calculate_makespan(processing_times, best_seq_pop)
        if best_cost_pop < best_cost:
            best_seq = best_seq_pop.copy()
            best_cost = best_cost_pop

    elapsed_time = time.time() - start_time

    return best_seq, best_cost, elapsed_time
