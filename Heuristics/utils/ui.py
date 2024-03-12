import streamlit as st
import numpy as np
import time
import pandas as pd


def generate_manual_input_grid(num_jobs, num_machines):
    # Creating a 2D array of placeholders to store the Streamlit input widgets
    input_data = [[0 for _ in range(num_machines)] for _ in range(num_jobs)]

    # Streamlit columns do not perfectly align like a traditional grid without some adjustments
    # Use the expander to make it visually appealing and organized
    with st.expander("Fill in the matrix"):
        # Generating the grid
        cols = st.columns(num_machines+1)  # Create a column for each machine
        for j, col in enumerate(cols):
            # Displaying the name of each machine at the top of each column
            if j == 0:
                col.write(f"J\M")
                continue
            col.write(f"Machine {j}")

            # Generating the grid
        for i in range(num_jobs+1):
            # Recreate a column for each machine

            for j in range(num_machines+1):
                # Using each column to take the input for each cell
                with cols[j]:
                    # Adding a label for each row on the first column
                    if j == 0:
                        st.text_input(
                            f"", value=f"Job {i}", key=f"job-{i}-machine-{j}")
                    else:
                        input_data[i-1][j-1] = int(st.text_input(
                            f"", value="0", key=f"job-{i}-machine-{j}"))

    return input_data


def display_output_array(output_data):
    st.write(np.array(output_data).reshape(-1, 1).T)


def display_matrix(matrix):
    # Determine the dimensions of the matrix
    num_rows, num_columns = matrix.shape

    # Generate column and row labels
    column_labels = [f"Machine {i+1}" for i in range(num_columns)]
    row_labels = [f"Job {i+1}" for i in range(num_rows)]

    # Create a DataFrame from the matrix with proper labeling
    df = pd.DataFrame(matrix, columns=column_labels, index=row_labels)

    # Display the DataFrame in Streamlit
    st.dataframe(df)


def generate_statistics_matrix():
    # Placeholder for algorithms and their statistics
    # Replace with your dynamic data retrieval or calculation
    algorithms = ["Algorithm 1", "Algorithm 2",
                  "Algorithm 3", "Algorithm 4", "Algorithm 5"]
    execution_times = np.random.rand(
        len(algorithms)) * 2  # Simulated execution times
    makespans = np.random.randint(
        100, 500, size=len(algorithms))  # Simulated makespans

    # Creating a DataFrame to hold the statistics
    statistics_df = pd.DataFrame({
        "Algorithm Name": algorithms,
        "Execution Time (s)": execution_times,
        "Makespan": makespans
    })

    return statistics_df
