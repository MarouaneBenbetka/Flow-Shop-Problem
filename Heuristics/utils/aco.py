import numpy as np
import random
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import calculate_makespan


class AntColonyOptimization:

    def __init__(self, processingTimes, Alpha=.5, Beta=.5, Q=.9, max_it=10, num_ant=16, rho=0, heuristicSolution=[]) -> None:

        self.numberJobs = processingTimes.shape[0]
        self.numberMachines = processingTimes.shape[1]
        self.Distances = np.zeros((self.numberJobs, self.numberJobs))
        self.processingTimes = processingTimes
        self.archive = heuristicSolution
        self.alpha = Alpha
        self.beta = Beta
        self.Q = Q
        self.globalPheromone = np.ones((self.numberJobs, self.numberJobs))
        self.maxIt = max_it
        self.numAnt = num_ant
        self.rho = rho

    def calculateDistances(self):
        for i in range(self.numberJobs):
            for j in range(self.numberJobs):
                if (i == j):
                    self.Distances[i, j] = 0
                else:
                    for k in range(self.numberMachines-1):
                        self.Distances[i, j] += self.processingTimes[j, k] + max(
                            0, self.processingTimes[i, k+1] - self.processingTimes[j, k])
                    self.Distances[i, j] += self.processingTimes[j,
                                                                 self.numberMachines-1]

    # calcule la formule de choix du job à prendre

    def calculateJobVoisin(self, jobCourant, lePotentielJobVoisin, lesPotentielsJobsVoisins):
        denominateur = 0
        numerateur = 0

        for i in range(len(lesPotentielsJobsVoisins)):
            denominateur += (self.globalPheromone[jobCourant, lesPotentielsJobsVoisins[i]])**self.alpha * (
                1/self.Distances[jobCourant, lesPotentielsJobsVoisins[i]])**self.beta
        numerateur = (self.globalPheromone[jobCourant, lePotentielJobVoisin])**self.alpha * (
            1/self.Distances[jobCourant, lePotentielJobVoisin])**self.beta

        # print("numerateur", numerateur)

        # print("denom", denominateur)

        return numerateur/denominateur

    def updateLocalPheromone(self, solutionSequence, localPheromoneMatrix):

        solutionQuality = calculate_makespan(
            self.processingTimes, solutionSequence)

        for j in range(len(solutionSequence)-1):

            localPheromoneMatrix[solutionSequence[j],
                                 solutionSequence[j+1]] += self.Q / solutionQuality

    def updateGlobalPheromone(self, localPheromoneMatrix):

        self.globalPheromone = (1-self.rho) * \
            self.globalPheromone + localPheromoneMatrix

    def run(self):

        start_time = time.time()

        self.calculateDistances()

        for it in range(self.maxIt):

            # liste de liste pour contenir les solutions de chaque fourmi

            solutions = []

            localPheromone = np.zeros((self.numberJobs, self.numberJobs))

            for ant in range(self.numAnt):

                solutions.append([])

                num_job_pris = 0

                # initialisation de la liste contenant les jobs qui constitueront la solution, elle sera updaté à chaque fois qu'un job est pris

                job_dispo = list(range(self.numberJobs))

                # démarrer par un job aléatoirement

                solutions[ant].append(
                    job_dispo[random.randint(0, len(job_dispo) - 1)])

                # updating ce qu'il faut

                job_dispo.remove(solutions[ant][num_job_pris])

                num_job_pris += 1

                while (len(job_dispo) > 0):  # équivaut à dire num_job_pris < self.numberJobs

                    bestVoisin = 0

                    nextJob = None

                    for i in range(len(job_dispo)):

                        # solutions[ant][num_job_pris-1] is the current node

                        voisinActuel = self.calculateJobVoisin(
                            solutions[ant][num_job_pris-1], job_dispo[i], job_dispo)

                        if (voisinActuel > bestVoisin):

                            bestVoisin = voisinActuel

                            nextJob = job_dispo[i]

                    solutions[ant].append(nextJob)

                    # updating ce qu'il faut

                    job_dispo.remove(nextJob)

                    num_job_pris += 1

                # Mise à jour local de phéromone

                self.updateLocalPheromone(solutions[ant], localPheromone)

                # print("Ant : ", ant, "'s solution : ", solutions[ant])

                # print("Its makespan is : ", calculate_makespan(self.processingTimes, solutions[ant]))

            # Mise à jour globale de phéromone

            self.updateGlobalPheromone(localPheromone)

            # L'idée que j'ai est de calculer le makespan de toutes les solutions et si une est meilleure de celle contenu dans l'archive l'archiver à son tour

            for sol in range(len(solutions)):

                makespan = calculate_makespan(
                    self.processingTimes, solutions[sol])

                if (makespan < calculate_makespan(self.processingTimes, self.archive)):

                    self.archive = solutions[sol]

        print("Elapsed time:", time.time()-start_time, "seconds")

        # Retourner la meilleure solution

        return self.archive
