# lstm_training.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import TensorBoard
import os
import datetime
import matplotlib.pyplot as plt
import joblib

def preprocess_data(input_file='position_temperature_data.npz', output_file='scalers_minmax.pkl'):
    data = np.load(input_file)['data']
    df = pd.DataFrame(data, columns=['time_index', 'feature_index', 'value'])
    df_pivot = df.pivot(index='time_index', columns='feature_index', values='value')

    scalers_minmax = {}
    for col in df_pivot.columns:
        scaler_minmax = MinMaxScaler()
        df_pivot[col] = scaler_minmax.fit_transform(df_pivot[[col]].values)
        scalers_minmax[col] = scaler_minmax
        print(f"Feature {col}: min={scaler_minmax.data_min_}, max={scaler_minmax.data_max_}")

    joblib.dump(scalers_minmax, output_file)
    df_pivot.fillna(method='ffill', inplace=True)
    df_pivot.fillna(method='bfill', inplace=True)
    return df_pivot.to_numpy(), scalers_minmax  # Return scalers_minmax along with data_array

def create_lstm_cnn_model(input_shape, output_units):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units=50))
    model.add(Dense(units=output_units))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_cnn(data_array, scalers_minmax, log_dir='logs/fit_minmax', model_file='lstm_cnn.keras'):
    window_size = 100
    lag_size = 10

    def create_sliding_windows(data_array, window_size, lag_size):
        X = []
        y = []
        for i in range(0, len(data_array) - window_size, lag_size):
            X.append(data_array[i:i + window_size])
            y.append(data_array[i + window_size])
        return np.array(X), np.array(y)

    X, y = create_sliding_windows(data_array, window_size, lag_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = create_lstm_cnn_model((X_train.shape[1], X_train.shape[2]), y_train.shape[1])
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    history = model.fit(X_train, y_train, epochs=200, batch_size=100, validation_split=0.2, callbacks=[tensorboard_callback])
    loss = model.evaluate(X_test, y_test)
    print(f'Test Loss (Min-Max Normalization): {loss}')

    y_pred = model.predict(X_test)
    model.save(model_file)

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss (Min-Max)')
    plt.plot(history.history['val_loss'], label='Validation Loss (Min-Max)')
    plt.title('Model Loss (Min-Max Normalization)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

    num_features = y_test.shape[1]
    num_plots_per_canvas = 6
    num_canvases = int(np.ceil(num_features / num_plots_per_canvas))

    for canvas_idx in range(num_canvases):
        plt.figure(figsize=(15, 10))
        for plot_idx in range(num_plots_per_canvas):
            feature_idx = canvas_idx * num_plots_per_canvas + plot_idx
            if feature_idx < num_features:
                plt.subplot(3, 2, plot_idx + 1)
                plt.plot(y_test[:, feature_idx], label='Actual')
                plt.plot(y_pred[:, feature_idx], label='Predicted')
                plt.title(f'Position {feature_idx} - Expected vs Actual (Min-Max Normalization)')
                plt.xlabel('Samples')
                plt.ylabel('Value')
                plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    print("Maximum values for each feature:")
    for i, (col, scaler) in enumerate(scalers_minmax.items()):
        print(f"Feature {i} ({col}): {scaler.data_max_[0]}")

if __name__ == "__main__":
    data_array, scalers_minmax = preprocess_data()
    train_lstm_cnn(data_array, scalers_minmax)  # Pass both arguments here
