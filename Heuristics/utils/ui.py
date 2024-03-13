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
    # Create a DataFrame from the output_data
    df = pd.DataFrame([output_data])
    # Adjust column names to start from 1
    new_columns = {i: f"Job {i+1}" for i in range(len(df.columns))}
    df.rename(columns=new_columns, inplace=True)
    # Set the row name to "Jobs"
    df.index = ["Jobs"]
    # Display the DataFrame
    st.write(df)


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
