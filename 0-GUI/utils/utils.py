import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import os


def calculate_makespan(processing_times, sequence):
    n_jobs = len(sequence)
    n_machines = len(processing_times[0])
    end_time = [[0] * (n_machines + 1) for _ in range(n_jobs + 1)]

    for j in range(1, n_jobs + 1):
        for m in range(1, n_machines + 1):
            end_time[j][m] = max(end_time[j][m - 1], end_time[j - 1]
                                 [m]) + processing_times[sequence[j - 1]][m - 1]

    return end_time[n_jobs][n_machines]


def generate_gantt_chart(processing_times, seq, interval=50, labeled=True):
    data = processing_times.T
    nb_jobs, nb_machines = processing_times.shape
    schedules = np.zeros((nb_machines, nb_jobs), dtype=dict)
    # schedule first job alone first
    task = {"name": "job_{}".format(
        seq[0]+1), "start_time": 0, "end_time": data[0][seq[0]]}

    schedules[0][0] = task
    for m_id in range(1, nb_machines):
        start_t = schedules[m_id-1][0]["end_time"]
        end_t = start_t + data[m_id][0]
        task = {"name": "job_{}".format(
            seq[0]+1), "start_time": start_t, "end_time": end_t}
        schedules[m_id][0] = task

    for index, job_id in enumerate(seq[1::]):
        start_t = schedules[0][index]["end_time"]
        end_t = start_t + data[0][job_id]
        task = {"name": "job_{}".format(
            job_id+1), "start_time": start_t, "end_time": end_t}
        schedules[0][index+1] = task
        for m_id in range(1, nb_machines):
            start_t = max(schedules[m_id][index]["end_time"],
                          schedules[m_id-1][index+1]["end_time"])
            end_t = start_t + data[m_id][job_id]
            task = {"name": "job_{}".format(
                job_id+1), "start_time": start_t, "end_time": end_t}
            schedules[m_id][index+1] = task

    # create a new figure
    fig, ax = plt.subplots(figsize=(18, 8))

    # set y-axis ticks and labels
    y_ticks = list(range(len(schedules)))
    y_labels = [f'Machine {i+1}' for i in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # calculate the total time
    total_time = max([job['end_time'] for proc in schedules for job in proc])

    # set x-axis limits and ticks
    ax.set_xlim(0, total_time)
    x_ticks = list(range(0, total_time+1, interval))
    ax.set_xticks(x_ticks)

    # set grid lines
    ax.grid(True, axis='x', linestyle='--')

    # create a color dictionary to map each job to a color
    color_dict = {}
    for proc in schedules:
        for job in proc:
            if job['name'] not in color_dict:
                color_dict[job['name']] = (np.random.uniform(
                    0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))

    # plot the bars for each job on each processor
    for i, proc in enumerate(schedules):
        for job in proc:
            start = job['start_time']
            end = job['end_time']
            duration = end - start
            color = color_dict[job['name']]
            ax.barh(i, duration, left=start, height=0.5,
                    align='center', color=color, alpha=0.8)
            if labeled:
                # add job labels
                label_x = start + duration/2
                label_y = i
                ax.text(
                    label_x, label_y, job['name'][4:], ha='center', va='center', fontsize=10)

    unique_id = datetime.now().strftime("%Y%m%d%H%M%S")

    folder = os.path.join(os.getcwd(), "0-GUI", "images", "gantt")

    filename = os.path.join(
        folder, f"gantt_chart_{unique_id}.png")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()  # Close the figure to prevent it from displaying in notebooks or IPython environments

    return filename


def generate_histogram(tab_stats):

    RDP = [stat["RDP"] for stat in tab_stats]
    Execution_time = [stat["Execution Time (ms)"] for stat in tab_stats]

    fig, ax = plt.subplots(figsize=(18, 10))

    index = np.arange(len(tab_stats))
    bar_width = 0.35

    rects1 = ax.bar(index, RDP, bar_width,
                    color='#D9731A',
                    label='RDP')

    rects2 = ax.bar(index + bar_width, Execution_time, bar_width,
                    color='#2A2359',
                    label='Execution Time')

    ax.set_xlabel('Heuristics')
    ax.set_ylabel('RDP / Execution Time (ms)')

    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels([stat['Algorithm'] for stat in tab_stats])

    ax.legend()

    unique_id = datetime.now().strftime("%Y%m%d%H%M%S")
    folder = os.path.join(os.getcwd(), "Heuristics", "images", "histograms")

    if not os.path.exists(folder):
        os.makedirs(folder)

    filename = os.path.join(folder, f"Histogram_{unique_id}.png")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()  # Close the figure to prevent it from displaying in notebooks or IPython environments

    return filename
