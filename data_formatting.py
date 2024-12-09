# data_formatting.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_position_data(data_array, position_index):
    time_indices = []
    temperatures = []

    for i in range(data_array.shape[0]):
        if int(data_array[i, 1]) == position_index:
            time_indices.append(data_array[i, 0])
            temperatures.append(data_array[i, 2])

    time_indices = np.array(time_indices, dtype=float)
    temperatures = np.array(temperatures, dtype=float)

    sorted_indices = np.argsort(time_indices)
    time_indices = time_indices[sorted_indices]
    temperatures = temperatures[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.plot(time_indices, temperatures, marker='o', linestyle='-', color='b')
    plt.xlabel('Time Index', fontsize=14)
    plt.ylabel('Temperature', fontsize=14)
    plt.title(f'Temperature vs. Time for Position {position_index}', fontsize=16)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.grid(True)
    plt.show()

def format_data(csv_directory='ship_data/', output_file='position_temperature_data.npz'):
    csv_files = [os.path.join(csv_directory, file) for file in os.listdir(csv_directory) if file.endswith('.csv')]
    data = []
    position_time_indices = {i: 0 for i in range(3, 4)}

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, encoding='latin1')
        for i in range(3, 4):
            position = df.iloc[i, 0]
            for j in range(1, 440):
                position_time_indices[i] += 1
                time_index = position_time_indices[i]
                temperature = df.iloc[i, j]
                position_index = i
                data.append((time_index, position_index, temperature))

    data_array = np.array(data)
    #for position_index in range(3, 4):
    #    plot_position_data(data_array, position_index)

    np.savez(output_file, data=data_array)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    format_data()
