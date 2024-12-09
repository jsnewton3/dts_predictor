import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import socket
import time

# Load the trained model and scalers
model = load_model('lstm_cnn.keras')
scalers_minmax = joblib.load('scalers_minmax.pkl')

# Define the window size and lag size
window_size = 100
lag_size = 1  # We'll update one step at a time

# Initialize data buffer
data_buffer = np.empty((0, len(scalers_minmax)))  # Adjust the shape according to the number of features

def preprocess_data(data, scalers_minmax):
    # Apply Min-Max scaling
    for col in data.columns:
        colu = col + 3
        scaler = scalers_minmax[str(colu)]
        data[col] = scaler.transform(data[[col]])
    return data

def inverse_transform_predictions(predictions, scalers_minmax):
    # Assume predictions is a 2D array with shape (n_samples, n_features)
    original_scale_predictions = np.zeros_like(predictions)
    for i in range(predictions.shape[1]):  # Assuming one feature, adjust if multiple
        feature_name = i + 3
        scaler = scalers_minmax[str(feature_name)]
        min_val = scaler.data_min_[0]
        max_val = scaler.data_max_[0]
        original_scale_predictions[:, i] = predictions[:, i] * (max_val - min_val) + min_val
    return original_scale_predictions

def inverse_transform_actual_values(data_array, scalers_minmax):
    # Assume data_array is a 2D array with shape (n_samples, n_features)
    original_scale_data = np.zeros_like(data_array)
    for i in range(data_array.shape[1]):  # Assuming one feature, adjust if multiple
        feature_name = i + 3
        scaler = scalers_minmax[str(feature_name)]
        min_val = scaler.data_min_[0]
        max_val = scaler.data_max_[0]
        original_scale_data[:, i] = data_array[:, i] * (max_val - min_val) + min_val
    return original_scale_data

def create_sliding_windows(data_array, window_size):
    X = []
    for i in range(len(data_array) - window_size + 1):
        X.append(data_array[i:i + window_size])
    return np.array(X)

def make_prediction(model, input_data):
    input_data = input_data.reshape(1, window_size, input_data.shape[1])  # Reshape for model input
    prediction = model.predict(input_data)
    return prediction

def add_data_to_buffer(new_data, buffer):
    return np.vstack([buffer, new_data])

def start_prediction_server(ip, port, output_file):
    # Set up UDP socket to receive data
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))

    global data_buffer

    # Open the output file in append mode
    with open(output_file, 'a') as f:
        print(f"Opened file {output_file} for writing.")
        try:
            while True:
                # Receive data from the streaming script
                data, addr = sock.recvfrom(1024)  # Buffer size is 1024 bytes
                values = list(map(float, data.decode().split(',')))

                # Create a DataFrame with the received data
                time_index = pd.Timestamp.now()
                feature_index = np.arange(len(values))
                new_data = pd.DataFrame({'time_index': [time_index]*len(values), 'feature_index': feature_index, 'value': values})
                new_data = new_data.pivot(index='time_index', columns='feature_index', values='value')
                print(f"New data received:\n{new_data}")

                # Preprocess the new data
                new_data = preprocess_data(new_data, scalers_minmax)
                new_data_array = new_data.to_numpy().reshape(1, -1)

                # Add new data to buffer
                data_buffer = add_data_to_buffer(new_data_array, data_buffer)

                # Check if buffer has enough data to form a window
                if len(data_buffer) >= window_size:
                    # Create sliding window from buffer
                    X_new = create_sliding_windows(data_buffer, window_size)

                    # Extract the actual values at the start of the relevant window
                    actual_values_start = data_buffer[-window_size].flatten()
                    # Reshape for inverse transformation
                    actual_values_start = actual_values_start.reshape(1, -1)

                    # Inverse transform the actual values
                    actual_values_start_original_scale = inverse_transform_actual_values(actual_values_start, scalers_minmax)

                    # Make prediction
                    prediction = make_prediction(model, X_new[-1])
                    prediction_original_scale = inverse_transform_predictions(prediction, scalers_minmax)

                    # Print actual values at the start of the window and the predicted value
                    print(f"Actual values at the start of the window (original scale): {actual_values_start_original_scale.flatten()}")
                    print(f"Prediction for latest window: {prediction_original_scale[0][0]}")

                    # Write the prediction result to the output file
                    #f.write(f"{time_index},{prediction_original_scale[0][0]}\n")
                    f.write(f"{time_index},{actual_values_start_original_scale.flatten()[0]},{prediction_original_scale[0][0]}\n")
                    f.flush()  # Ensure data is written to disk

                    # Optionally, remove the oldest data point to keep the buffer size constant
                    data_buffer = data_buffer[-window_size:]

                time.sleep(1)  # Sleep for a second before the next data point

        except KeyboardInterrupt:
            print("Real-time prediction stopped.")
            sock.close()

# Add this line if you want to run this script standalone
if __name__ == "__main__":
    start_prediction_server("127.0.0.1", 5005, "predictions_output.csv")
