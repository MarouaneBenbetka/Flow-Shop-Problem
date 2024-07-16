# Flow Shop Problem Optimization

This repository contains an application of various optimization techniques to solve the Flow Shop Scheduling Problem (FSSP). It encompasses multiple approaches, from brute force solutions using Branch and Bound to advanced methods like Metaheuristics and a custom solution named FSSP-DQN. The repository is structured to guide users through different optimization methodologies and integrates all solutions into a single web application for ease of use.

## Flow Shop Scheduling Problem (FSSP)

The Flow Shop Scheduling Problem is a classic optimization problem where a set of jobs must be processed on a series of machines in a specific order. The objective is to minimize the total processing time, also known as makespan. This problem is NP-hard, making it challenging to find optimal solutions for large instances.

## Proposed Solutions

### 1. Branch and Bound

The `1-Branch&Bound` directory contains a brute force solution using the Branch and Bound method. This method systematically explores all possible sequences of jobs, pruning branches that cannot yield better solutions. The `branch_and_bound.ipynb` notebook demonstrates how this method is applied to solve FSSP using problem instances from the `data` directory.

### 2. Heuristics

The `2-Heuristics` directory includes heuristic methods for solving FSSP. Heuristics provide approximate solutions by using rules of thumb or intuitive methods. The `heuristics.ipynb` notebook in the `utils` folder provides implementations of various heuristic algorithms, which are faster but may not always produce the optimal solution.

### 3. MetaHeuristics

This section is divided into two parts:

-   `3.1-MetaHeuristics`: Focuses on neighborhood-based metaheuristic algorithms, such as Tabu Search. These algorithms explore the solution space by moving from one solution to a neighboring solution. The `metaheuristics_voisinage.ipynb` notebook in the `utils` folder contains implementations of these algorithms.
-   `3.2-MetaHeuristics`: Covers population-based metaheuristic algorithms, such as Genetic Algorithms and Particle Swarm Optimization. These methods use a population of solutions to explore the solution space and apply operators like selection, crossover, and mutation. The `population_based_metaheuristics.ipynb` notebook in the `utils` folder demonstrates these methods.

### 4. FSSP-DQN

The `4-FSSP-DQN` directory presents a custom solution named FSSP-DQN, which is designed to find the best hyperparameters for a given set of metaheuristics. It uses Deep Q-Networks (DQN) to learn and optimize the hyperparameters dynamically. The `fssp_dqn.ipynb` notebook explains the implementation and application of this approach.

For more details, refer to our [paper on FSSP-DQN](./github/Deep_Q_Networks_for_Hyperparameter_Tuning_in_Flow_Shop_Scheduling_Problem.pdf).

### 0-GUI

The `0-GUI` directory contains a Streamlit application that integrates all the solutions into a single web interface for ease of use. This application allows users to interact with the various optimization techniques through a user-friendly website.

## Benchmark Testing

The algorithms were tested using the E. Taillard benchmarks, which can be found at [E. Taillard Benchmark Website](http://mistic.heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/ordonnancement.html).

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/fssp-optimization.git
    cd fssp-optimization
    ```
2. Navigate to the desired optimization method directory and open the corresponding Jupyter notebook to explore the implementations.

3. To run the Streamlit application, navigate to the `0-GUI` directory and execute:
    ```bash
    streamlit run main.py
    ```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.
