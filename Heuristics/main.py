import streamlit as st
import numpy as np
import time
import pandas as pd
from utils.ui import *
from utils.heurstics import neh_algorithm, ham_heuristic, cds_heuristic, gupta_heuristic, run_palmer, PRSKE, special_heuristic, NRH, chen_heuristic
from utils.benchmarks import benchmarks
from utils.utils import generate_gantt_chart, generate_histogram
import os
# Placeholder for your algorithm execution function
# This should return a 1D array, execution time, and an image path


def run_algorithm(algo, input_data):
    start_time = time.perf_counter()
    output_data, makespan = algo["algo"](input_data)
    end_time = time.perf_counter()
    execution_time_micros = (end_time - start_time) * 1e3

    image_path = generate_gantt_chart(input_data, output_data)
    # Placeholder image path
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
            "RDP": f'{round((makespan - benchmark_data["upper-bound"]) / (benchmark_data["upper-bound"]) * 100,2)}%'
        })
    image_path = generate_histogram(stats)
    return pd.DataFrame(sorted(stats, key=lambda x: x["Makespan"])), image_path


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


def main():

    print(os.getcwd())
    st.title("Algorithm Showcase")

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


if __name__ == "__main__":
    main()
