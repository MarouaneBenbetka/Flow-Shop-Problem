import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime


def calculate_makespan(processing_times, sequence):
    n_jobs = len(sequence)
    n_machines = len(processing_times[0])
    end_time = [[0] * (n_machines + 1) for _ in range(n_jobs + 1)]

    for j in range(1, n_jobs + 1):
        for m in range(1, n_machines + 1):
            end_time[j][m] = max(end_time[j][m - 1], end_time[j - 1]
                                 [m]) + processing_times[sequence[j - 1]][m - 1]

    return end_time[n_jobs][n_machines]


def generate_gantt_chart(current_instance, solution):
    plt.figure(figsize=(20, 12))
    df = pd.DataFrame(columns=['Machine', 'Job', 'Start', 'Finish'])

    machines, jobs = current_instance.shape
    start_time_m = np.zeros(machines)
    for job in solution:

        for machine_index in range(machines):
            start_time = start_time_m[machine_index]
            if machine_index > 0:
                start_time = max(start_time, start_time_m[machine_index-1])
            end_time = start_time + current_instance[machine_index, job]
            start_time_m[machine_index] = end_time

            df = pd.concat([df, pd.DataFrame({'Machine': f'Machine {machine_index + 1}',
                                              'Job': f'Job {job + 1}',
                                              'Start': start_time,
                                              'Finish': end_time}, index=[0])], ignore_index=True)

    colors = plt.cm.tab10.colors
    for i, machine_index in enumerate(range(machines)):
        machine_df = df[df['Machine'] == f'Machine {machine_index + 1}']
        plt.broken_barh([(start, end - start) for start, end in zip(machine_df['Start'], machine_df['Finish'])],
                        (i * 10, 9), facecolors=[colors[j % 10] for j in range(jobs)], edgecolor='black')

    plt.xlabel('Time')
    plt.yticks([i * 10 + 4.5 for i in range(machines)],
               [f'Machine {i + 1}' for i in range(machines)])

    # Generate a unique ID for the file based on the current timestamp
    unique_id = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"images/gantt_chart_{unique_id}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()  # Close the figure to prevent it from displaying in notebooks or IPython environments

    return filename
