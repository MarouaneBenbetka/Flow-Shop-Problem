import streamlit as st
import numpy as np
import time
import pandas as pd
from utils.ui import *

# Placeholder for your algorithm execution function
# This should return a 1D array, execution time, and an image path


def run_algorithm(algo_name, input_data):
    start_time = time.time()
    # Simulate processing with a sleep
    time.sleep(np.random.rand())
    output_data = np.random.randint(0, 100, size=25).tolist()  # Example output
    execution_time = time.time() - start_time
    # Example count (you can adjust this to be whatever metric you need)
    count = len(output_data)
    # Placeholder image path
    image_path = "./images/Screenshot 2024-03-01 212709.png"
    return output_data, execution_time, count, image_path


# Define your algorithms names
algorithms = [
    "Algorithm 1",
    "Algorithm 2",
    "Algorithm 3",
    "Algorithm 4",
    "Algorithm 5",
    "Algorithm 6",
    "Algorithm 7",
    "Algorithm 8",
    "Algorithm 9"
]

# List of benchmarks for demonstration
benchmarks = {
    "Benchmark 1": np.random.rand(5, 5),
    "Benchmark 2": np.random.rand(7, 7),
    # Add your actual benchmarks here
}


def main():
    st.title("Algorithm Showcase")

    # Algorithm selection
    st.header("Select an Algorithm")
    cols = st.columns(3)
    selected_algorithm = st.session_state.get('selected_algorithm', None)
    for index, algorithm in enumerate(algorithms):
        with cols[index % 3]:
            if st.button(algorithm, key=algorithm, args=(algorithm,)):
                selected_algorithm = algorithm
                st.session_state['selected_algorithm'] = algorithm

    if selected_algorithm:
        st.header(f"Configurations for {selected_algorithm}")
        option = st.radio(
            "Input Method", ["Manual", "Generate Random", "Benchmark"])

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
                "Choose a Benchmark", list(benchmarks.keys()))
            input_data = benchmarks[benchmark_selection]
            display_matrix(input_data)

        if st.button("Run Algorithm"):
            output_data, execution_time, count, image_path = run_algorithm(
                selected_algorithm, input_data)

            st.subheader("Results")
            # Displaying the output matrix directly
            st.write("Output Matrix:")
            st.write(output_data)
            st.write(f"Execution Time: {execution_time:.2f} seconds")
            st.write(f"Count: {count}")
            st.image(image_path, caption="Output Image")

    if st.button("📊 Show General Statistics", help="Click to view general statistics about the algorithms"):
        # Generate the statistics matrix
        statistics_df = generate_statistics_matrix()

        st.subheader("General Statistics")
        st.table(statistics_df)


if __name__ == "__main__":
    main()
