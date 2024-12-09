# main.py

from data_formatting import format_data
from lstm_training import preprocess_data, train_lstm_cnn
from rearrange import rearrange_data
from rearrange_streaming import stream_data
import predictor3
import threading
import subprocess

def start_streaming():
    # Rearrange data and start streaming
    data_array, output_csv_file, output_npz_file = rearrange_data()
    stream_data(output_npz_file, "127.0.0.1", 5005)

def start_prediction():
    # Define the output file for predictions
    output_prediction_file = "predictions_output.csv"
    
    # Start the prediction server with the output file
    predictor3.start_prediction_server("127.0.0.1", 5005, output_prediction_file)

def start_plotting():
    # Start the plotting script
    subprocess.Popen(['python3', 'plot_predictions.py'])

def main():
    # Step 1: Format the data
    print("Formatting data...")
    format_data()
    
    # Step 2: Train the LSTM model
    print("Preprocessing data for LSTM...")
    data_array, scalers_minmax = preprocess_data()
    
    print("Training LSTM model...")
    train_lstm_cnn(data_array, scalers_minmax)

    # Create threads for streaming, prediction, and plotting
    streaming_thread = threading.Thread(target=start_streaming)
    prediction_thread = threading.Thread(target=start_prediction)
    plotting_thread = threading.Thread(target=start_plotting)
    
    # Start threads
    streaming_thread.start()
    prediction_thread.start()
    plotting_thread.start()
    
    # Join threads to wait for their completion
    streaming_thread.join()
    prediction_thread.join()
    plotting_thread.join()

if __name__ == "__main__":
    main()
