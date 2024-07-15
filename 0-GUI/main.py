import streamlit as st
import numpy as np
import time
import pandas as pd
from utils.ui import *
from utils.heurstics import neh_algorithm, ham_heuristic, cds_heuristic, gupta_heuristic, run_palmer, PRSKE, special_heuristic, NRH, chen_heuristic
from utils.benchmarks import benchmarks
from utils.utils import generate_gantt_chart, generate_histogram, calculate_makespan
import os
from utils.rs import RS, RS_fba
from utils.aco import AntColonyOptimization
from utils.abc import ArtificialBeeColony
from utils.ag import genetic_algorithm
# Placeholder for your algorithm execution function
# This should return a 1D array, execution time, and an image path


def run_algorithm(algo, input_data, type="heuristique"):
    if type == "heuristique":
        start_time = time.perf_counter()
        output_data, makespan = algo["algo"](input_data)
        end_time = time.perf_counter()
        execution_time_micros = (end_time - start_time) * 1e3

        image_path = generate_gantt_chart(input_data, output_data)
    elif type == "RS" or type == "ACO":
        start_time = time.perf_counter()
        output_data = algo(input_data)
        print(output_data)

        makespan = calculate_makespan(input_data, output_data)
        end_time = time.perf_counter()
        execution_time_micros = (end_time - start_time) * 1e3

        image_path = generate_gantt_chart(input_data, output_data)
    elif type == 'AG':
        start_time = time.perf_counter()
        output_data = algo(input_data)
        print(output_data)

        makespan = calculate_makespan(input_data, output_data)
        end_time = time.perf_counter()
        execution_time_micros = (end_time - start_time) * 1e3

        image_path = generate_gantt_chart(input_data, output_data)

    return output_data, execution_time_micros, makespan, image_path


def generate_statistics(benchmark_data):
    stats = []
    for algorithm in algorithms:
        output_data, execution_time, makespan, _ = run_algorithm(
            algorithm, benchmark_data["data"])
        stats.append({
            "Algorithm": algorithm['name'],
            "Execution Time (ms)": execution_time,
            "Makespan": makespan,
            "RDP": round((makespan - benchmark_data["upper-bound"]) / (benchmark_data["upper-bound"]) * 100, 2)
        })

    sorted_stats = sorted(stats, key=lambda x: x["Makespan"])
    image_path = generate_histogram(sorted_stats)
    return pd.DataFrame(sorted_stats), image_path


# Define your algorithms names
algorithms = [
    {
        "name": "NEH",
        "algo":  neh_algorithm
    },
    {
        "name": "Ham",
        "algo":  ham_heuristic
    },
    {
        "name": "CDS",
        "algo":  cds_heuristic
    },
    {
        "name": "Gupta",
        "algo":  gupta_heuristic
    },
    {
        "name": "Palmer",
        "algo":  run_palmer
    },
    {
        "name": "PRSKE",
        "algo":  PRSKE
    },
    {
        "name": "Weighted CDS",
        "algo": special_heuristic
    },
    {
        "name": "NRH",
        "algo": NRH
    },
    {
        "name": "Chen",
        "algo": chen_heuristic
    }


]

# List of benchmarks for demonstration
benchmarks_list = {f"Instance {i+1}": b for i, b in enumerate(benchmarks)}


def heuristique_interface():

    # Algorithm selection
    st.header("Select an Algorithm")
    cols = st.columns(3)
    st.markdown("""
        <style>
        button[kind="secondary"] {
            display: inline-block;
            width: 100%;  # Makes the button fill the column width
            padding: 20px;  # Adjust the padding as needed
        }
        </style>
    """, unsafe_allow_html=True)

    if (st.button("📊 Show General Statistics", type="secondary")):
        # Toggle visibility of the statistics and benchmark selection
        st.session_state.show_statistics = not st.session_state.get(
            'show_statistics', False)
        selected_algorithm = None
        st.session_state['selected_algorithm'] = None

    selected_algorithm = st.session_state.get('selected_algorithm', None)
    for index, algorithm in enumerate(algorithms):
        name, model = algorithm['name'], algorithm['algo']
        with cols[index % 3]:
            if st.button(name, key=name, args=(name,), type="secondary"):
                selected_algorithm = algorithm
                st.session_state['selected_algorithm'] = algorithm

    if selected_algorithm:
        st.header(f"Configurations for {selected_algorithm['name']}")
        option = st.radio(
            "Input Method", ["Benchmark", "Manual", "Generate Random",])

        if option == "Manual":
            num_jobs = st.number_input(
                "Number of Jobs (lines)", min_value=1, value=5, key='man_jobs')
            num_machines = st.number_input(
                "Number of Machines (columns)", min_value=1, value=5, key='man_machines')

            # Generate manual input grid
            input_data = generate_manual_input_grid(num_jobs, num_machines)

        elif option == "Generate Random":
            num_jobs = st.number_input(
                "Number of Jobs (lines)", min_value=1, value=5, key='rand_jobs')
            num_machines = st.number_input(
                "Number of Machines (columns)", min_value=1, value=5, key='rand_machines')
            input_data = np.random.randint(
                1, 100, size=(num_jobs, num_machines))
            display_matrix(input_data)

        elif option == "Benchmark":
            benchmark_selection = st.selectbox(
                "Choose a Benchmark", list(benchmarks_list.keys()))
            input_data = benchmarks_list[benchmark_selection]["data"]
            display_matrix(input_data)

        if st.button("Run Algorithm"):
            output_data, execution_time, makespan, image_path = run_algorithm(
                selected_algorithm, input_data)

            st.subheader("Results")
            # Displaying the output matrix directly
            st.write("Output Matrix:")
            display_output_array(output_data)
            st.write(f"Execution Time: {execution_time:.4f} ms")
            st.write(f"Makespan: {makespan}")
            st.write(
                f"Upper-bound: {benchmarks_list[benchmark_selection]['upper-bound']}")
            st.image(image_path, caption="Output Image")

        # Initial setup for session state to manage the display of the stats and benchmark selection

    if not selected_algorithm and st.session_state.get('show_statistics', False):
        benchmark_selection = st.selectbox("Choose a Benchmark for Statistics", list(
            benchmarks_list.keys()), key='benchmark_selection')
        display_matrix(benchmarks_list[benchmark_selection]["data"])
        if benchmark_selection:
            with st.spinner('Calculating statistics... Please wait.'):
                # Generate and display statistics for the selected benchmark
                benchmark_data = benchmarks_list[benchmark_selection]

                statistics_df, image = generate_statistics(benchmark_data)

                st.subheader("General Statistics")
                st.write(
                    f"Upper-bound: {benchmarks_list[benchmark_selection]['upper-bound']}")
                st.table(statistics_df)
                st.image(image, caption="General statistics graph")


# Assuming all your other imports and functions are defined as before...


def main():
    st.title("Flow Shop Problem")

    # Navigation
    st.sidebar.title('Navigation')
    page = st.sidebar.radio(
        "Select a page:", ["Heuristiques", "Méthodes de voisinage", "ACO", "Artificial Bee Colony", "Algorithme Génétique"])

    if page == "Heuristiques":
        # All your existing code related to the Algorithm Showcase goes here
        st.header("Heuristiques")

        heuristique_interface()
    elif page == "Méthodes de voisinage":
        # Code for another page
        st.header("Méthodes de voisinage")
        run_simulated_annealing_page()
    elif page == 'ACO':
        st.header('ACO')
        run_ACO_page()
    elif page == 'Artificial Bee Colony':
        st.header('Artificial Bee Colony')
        run_ABC_page()
    elif page == 'Algorithme Génétique':
        st.header('Algorithme Génétique')
        run_AG_page()

# =====================


def generate_initial_solution(method, data):
    if method == "Manual":
        # Assume manual entry is already a complete solution
        return data
    else:
        # Use one of the heuristics to generate an initial solution
        return method(data)[0]


def run_simulated_annealing_page():
    st.header("Simulated Annealing Configuration")

    option = st.radio(
        "Input Method", ["Benchmark", "Manual", "Generate Random",])

    input_data = np.random.randint(
        1, 100, size=(5, 5))
    if option == "Manual":
        num_jobs = st.number_input(
            "Number of Jobs (lines)", min_value=1, value=5, key='man_jobs')
        num_machines = st.number_input(
            "Number of Machines (columns)", min_value=1, value=5, key='man_machines')

        # Generate manual input grid
        input_data = generate_manual_input_grid(num_jobs, num_machines)

    elif option == "Generate Random":
        num_jobs = st.number_input(
            "Number of Jobs (lines)", min_value=1, value=5, key='rand_jobs')
        num_machines = st.number_input(
            "Number of Machines (columns)", min_value=1, value=5, key='rand_machines')
        input_data = np.random.randint(
            1, 100, size=(num_jobs, num_machines))
        display_matrix(input_data)

    elif option == "Benchmark":
        benchmark_selection = st.selectbox(
            "Choose a Benchmark", list(benchmarks_list.keys()))
        input_data = benchmarks_list[benchmark_selection]["data"]
        display_matrix(input_data)

    # Initial solution selection
    init_sol_method = st.selectbox(
        "Select method for initial solution",
        ["NEH", "Ham", "CDS", "Gupta", "Palmer",
            "PRSKE", "Weighted CDS", "NRH", "Chen"]
    )

    algo = [a for a in algorithms if a['name'] == init_sol_method]
    # If manual, let user enter their solution
    if not algo:
        initial_solution = st.text_area("Enter your initial solution:", "")
    else:
        # Use a heuristic method
        initial_solution = generate_initial_solution(
            algo[0]['algo'], input_data)

    # Parameters for Simulated Annealing
    temp = st.number_input("Initial temperature", value=10.0, format="%.2f")
    alpha = st.slider("Alpha", 0.1, 0.99, 0.6, 0.05)
    nb_palier = st.number_input("Number of plateaus", value=1, step=1)
    it_max = st.number_input("Maximum iterations", value=100, step=10)

    swap_method = st.selectbox(
        "Swap method",
        ["random_swap", "best_swap", "first_admissible_swap",
            "first_best_adimissible_swap"]
    )

    if st.button("Run Simulated Annealing"):
        # Assuming the initial solution is parsed correctly

        if swap_method == "first_best_adimissible_swap":
            def main_algorithm(x): return RS_fba(
                x, initial_solution, temp, alpha, nb_palier, it_max)

        else:
            def main_algorithm(x): return RS(x, initial_solution, temp,
                                             swap_method, alpha, nb_palier, it_max)

        output_data, execution_time, makespan, image_path = run_algorithm(
            main_algorithm, input_data, type="RS")

        st.subheader("Results")
        # Displaying the output matrix directly
        st.write("Output Matrix:")
        display_output_array(output_data)
        st.write(f"Execution Time: {execution_time:.4f} ms")
        st.write(f"Makespan: {makespan}")
        st.write(
            f"Upper-bound: {benchmarks_list[benchmark_selection]['upper-bound']}")
        st.image(image_path, caption="Output Image")


def run_ACO_page():
    st.header("ACO Configuration")

    option = st.radio(
        "Input Method", ["Benchmark", "Manual", "Generate Random",])

    input_data = np.random.randint(
        1, 100, size=(5, 5))
    if option == "Manual":
        num_jobs = st.number_input(
            "Number of Jobs (lines)", min_value=1, value=5, key='man_jobs')
        num_machines = st.number_input(
            "Number of Machines (columns)", min_value=1, value=5, key='man_machines')

        # Generate manual input grid
        input_data = generate_manual_input_grid(num_jobs, num_machines)

    elif option == "Generate Random":
        num_jobs = st.number_input(
            "Number of Jobs (lines)", min_value=1, value=5, key='rand_jobs')
        num_machines = st.number_input(
            "Number of Machines (columns)", min_value=1, value=5, key='rand_machines')
        input_data = np.random.randint(
            1, 100, size=(num_jobs, num_machines))
        display_matrix(input_data)

    elif option == "Benchmark":
        benchmark_selection = st.selectbox(
            "Choose a Benchmark", list(benchmarks_list.keys()))
        input_data = benchmarks_list[benchmark_selection]["data"]
        display_matrix(input_data)

    # Initial solution selection
    init_sol_method = st.selectbox(
        "Select method for initial solution",
        ["NEH", "Ham", "CDS", "Gupta", "Palmer",
            "PRSKE", "Weighted CDS", "NRH", "Chen"]
    )

    algo = [a for a in algorithms if a['name'] == init_sol_method]
    # If manual, let user enter their solution
    if not algo:
        initial_solution = st.text_area("Enter your initial solution:", "")
    else:
        # Use a heuristic method
        initial_solution = generate_initial_solution(
            algo[0]['algo'], input_data)

    # Parameters for Simulated Annealing
    alpha = st.slider("Alpha", 0.1, 0.99, 0.6, 0.05)
    beta = st.slider("Beta", 0.1, 0.99, 0.6, 0.05)
    q = st.slider("Q", 0.1, 0.99, 0.6, 0.05)
    it_max = st.number_input("Maximum iterations", value=100, step=10)
    max_ant = st.number_input("num_ant", value=10, step=1)
    rho = st.slider("RHO", 0.1, 0.99, 0.6, 0.05)

    ACO = AntColonyOptimization(input_data, Alpha=alpha, Beta=beta, Q=q,
                                max_it=it_max, num_ant=max_ant, rho=rho, heuristicSolution=initial_solution)

    result = ACO.run()
    if st.button("Run ACO"):
        def main_algorithm(x): return ACO.run()

        output_data, execution_time, makespan, image_path = run_algorithm(
            main_algorithm, input_data, type="ACO")

        st.subheader("Results")
        # Displaying the output matrix directly
        st.write("Output Matrix:")
        display_output_array(output_data)
        st.write(f"Execution Time: {execution_time:.4f} ms")
        st.write(f"Makespan: {makespan}")
        if option == 'Benchmark':
            st.write(
                f"Upper-bound: {benchmarks_list[benchmark_selection]['upper-bound']}")
        st.image(image_path, caption="Output Image")


def run_ABC_page():
    st.header("Artificial Bee Colony Configuration")

    option = st.radio(
        "Input Method", ["Benchmark", "Manual", "Generate Random",])

    input_data = np.random.randint(
        1, 100, size=(5, 5))
    if option == "Manual":
        num_jobs = st.number_input(
            "Number of Jobs (lines)", min_value=1, value=5, key='man_jobs')
        num_machines = st.number_input(
            "Number of Machines (columns)", min_value=1, value=5, key='man_machines')

        # Generate manual input grid
        input_data = generate_manual_input_grid(num_jobs, num_machines)

    elif option == "Generate Random":
        num_jobs = st.number_input(
            "Number of Jobs (lines)", min_value=1, value=5, key='rand_jobs')
        num_machines = st.number_input(
            "Number of Machines (columns)", min_value=1, value=5, key='rand_machines')
        input_data = np.random.randint(
            1, 100, size=(num_jobs, num_machines))
        display_matrix(input_data)

    elif option == "Benchmark":
        benchmark_selection = st.selectbox(
            "Choose a Benchmark", list(benchmarks_list.keys()))
        input_data = benchmarks_list[benchmark_selection]["data"]
        display_matrix(input_data)

    # Initial solution selection
    init_sol_method = st.selectbox(
        "Select method for initial solution",
        ["NEH", "Ham", "CDS", "Gupta", "Palmer",
            "PRSKE", "Weighted CDS", "NRH", "Chen"]
    )

    # algo = [a for a in algorithms if a['name'] == init_sol_method]
    # # If manual, let user enter their solution
    # if not algo:
    #     initial_solution = st.text_area("Enter your initial solution:", "")
    # else:
    #     # Use a heuristic method
    #     initial_solution = generate_initial_solution(
    #         algo[0]['algo'], input_data)

    # Parameters for Simulated Annealing

    nPop = st.number_input("Number of Employees Bees", value=100, step=10)
    nOnLooker = st.number_input("Number of onlooker Bees", value=40, step=10)
    l = st.number_input("L", value=5, step=1)

    it_max = st.number_input("Maximum iterations", value=100, step=10)

    ABC = ArtificialBeeColony(
        input_data,  MaxIt=it_max, nPop=nPop, nOnlooker=nOnLooker, L=l)

    if st.button("Run ABC"):
        def main_algorithm(x): return ABC.run()

        output_data, execution_time, makespan, image_path = run_algorithm(
            main_algorithm, input_data, type="ACO")

        st.subheader("Results")
        # Displaying the output matrix directly
        st.write("Output Matrix:")
        display_output_array(output_data)
        st.write(f"Execution Time: {execution_time:.4f} ms")
        st.write(f"Makespan: {makespan}")
        if option == 'Benchmark':
            st.write(
                f"Upper-bound: {benchmarks_list[benchmark_selection]['upper-bound']}")
        st.image(image_path, caption="Output Image")


def run_AG_page():
    st.header("AG Configuration")

    option = st.radio(
        "Input Method", ["Benchmark", "Manual", "Generate Random",])

    input_data = np.random.randint(
        1, 100, size=(5, 5))
    if option == "Manual":
        num_jobs = st.number_input(
            "Number of Jobs (lines)", min_value=1, value=5, key='man_jobs')
        num_machines = st.number_input(
            "Number of Machines (columns)", min_value=1, value=5, key='man_machines')

        # Generate manual input grid
        input_data = generate_manual_input_grid(num_jobs, num_machines)

    elif option == "Generate Random":
        num_jobs = st.number_input(
            "Number of Jobs (lines)", min_value=1, value=5, key='rand_jobs')
        num_machines = st.number_input(
            "Number of Machines (columns)", min_value=1, value=5, key='rand_machines')
        input_data = np.random.randint(
            1, 100, size=(num_jobs, num_machines))
        display_matrix(input_data)

    elif option == "Benchmark":
        benchmark_selection = st.selectbox(
            "Choose a Benchmark", list(benchmarks_list.keys()))
        input_data = benchmarks_list[benchmark_selection]["data"]
        display_matrix(input_data)

    # Initial solution selection
    init_sol_method = st.selectbox(
        "Select method for initial solution",
        ["NEH", "Ham", "CDS", "Gupta", "Palmer",
            "PRSKE", "Weighted CDS", "NRH", "Chen"]
    )

    algo = [a for a in algorithms if a['name'] == init_sol_method]
    # If manual, let user enter their solution
    if not algo:
        initial_solution = st.text_area("Enter your initial solution:", "")
    else:
        # Use a heuristic method
        initial_solution = generate_initial_solution(
            algo[0]['algo'], input_data)

    # processing_times, init_pop, pop_size, select_pop_size, selection_method, crossover, mutation_probability, num_iterations

    # Parameters for Simulated Annealing
    pop_size = st.number_input("Pop size", value=30, step=5)
    select_pop_size = st.slider("select_pop_size", 0.1, 0.99, 0.6, 0.05)
    mutation_probability = st.slider(
        "Mutation Probability", 0.1, 0.99, 0.6, 0.05)

    it_max = st.number_input("Maximum iterations", value=100, step=10)

    selection_method = st.selectbox(
        "Selection Method",
        ["roulette", "Elitism", "rank", "tournament"]
    )

    crossover = st.selectbox(
        "Crossover",
        ["ONE", "TWO"]
    )

    if st.button("Run AG"):
        def main_algorithm(x): return genetic_algorithm(
            x, initial_solution, pop_size, select_pop_size, selection_method, crossover, mutation_probability, it_max)[0]

        output_data, execution_time, makespan, image_path = run_algorithm(
            main_algorithm, input_data, type="AG")

        st.subheader("Results")
        # Displaying the output matrix directly
        st.write("Output Matrix:")
        display_output_array(output_data)
        st.write(f"Execution Time: {execution_time:.4f} ms")
        st.write(f"Makespan: {makespan}")
        if option == 'Benchmark':
            st.write(
                f"Upper-bound: {benchmarks_list[benchmark_selection]['upper-bound']}")
        st.image(image_path, caption="Output Image")


# This line is needed to run the application
if __name__ == "__main__":
    main()
