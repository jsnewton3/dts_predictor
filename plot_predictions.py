import matplotlib.pyplot as plt
import pandas as pd
import time
import os

def plot_predictions(file_path):
    plt.ion()  # Turn on interactive mode for real-time plotting
    fig, ax = plt.subplots()
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature')
    ax.set_title('Temperature vs Time')

    while True:
        if os.path.exists(file_path):
            # Read the file with both actual and predicted values
            data = pd.read_csv(file_path, names=['time_index', 'actual', 'prediction'], header=None)
            
            # Ensure there are enough rows to plot
            if len(data) > 1:
                # Convert 'time_index' to datetime
                data['time_index'] = pd.to_datetime(data['time_index'])
                
                # Clear the previous plot
                ax.clear()
                
                # Plot actual and predicted values
                ax.plot(data['time_index'], data['actual'], label='Actual Values', color='blue')
                ax.plot(data['time_index'], data['prediction'], label='Predicted Values', color='red')
                
                ax.set_xlabel('Time')
                ax.set_ylabel('Temperature')
                ax.set_title('Temperature vs Time')
                ax.legend()
                
                plt.draw()
                
            plt.pause(1)  # Pause for 1 second before updating the plot

        time.sleep(1)  # Check the file every second

if __name__ == "__main__":
    plot_predictions('predictions_output.csv')
