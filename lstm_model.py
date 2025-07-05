import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# TensorFlow and Keras imports for the LSTM model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping


def create_sequences(X_data, y_data, time_steps=60):
    """
    Converts time series data into sequences for LSTM model training.
    To predict the value at time t, it uses data from t-time_steps to t-1.
    """
    Xs, ys = [], []
    for i in range(len(X_data) - time_steps):
        Xs.append(X_data[i:(i + time_steps)])
        ys.append(y_data[i + time_steps])
    return np.array(Xs), np.array(ys)


class LSTMModel:
    """
    A class to encapsulate the LSTM model for time series prediction.
    It handles data reshaping, model building, training, and evaluation.
    """

    def __init__(self, time_steps=60):
        """
        Initializes the LSTMModel.

        Args:
            time_steps (int): The number of past time steps to use for prediction.
        """
        self.time_steps = time_steps
        self.model = None
        self.is_trained = False
        print(f"LSTM model initialized with time_steps={self.time_steps}.")

    def build_model(self, input_shape):
        """
        Builds the LSTM model architecture.
        """
        model = Sequential()
        # First LSTM layer with Dropout regularization
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        # Second LSTM layer
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        # Dense layer
        model.add(Dense(units=25))
        # Output layer
        model.add(Dense(units=1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        print("LSTM model built successfully.")
        self.model.summary()

    def train(self, X_train_scaled, y_train_scaled, epochs=50, batch_size=32):
        """
        Trains the LSTM model.

        Args:
            X_train_scaled (np.ndarray): The scaled training features.
            y_train_scaled (np.ndarray): The scaled training target values.
            epochs (int): The number of training epochs.
            batch_size (int): The size of batches for training.
        """
        print("Preparing data into sequences...")
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, self.time_steps)

        if X_train_seq.shape[0] == 0:
            raise ValueError("Not enough data to create sequences with the given time_steps.")

        print(f"Sequence shapes: X_train_seq={X_train_seq.shape}, y_train_seq={y_train_seq.shape}")

        # Build the model based on the input shape
        self.build_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))

        print("\nTraining the LSTM model... (This can take a significant amount of time)")

        # Use EarlyStopping to prevent overfitting
        early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1)

        history = self.model.fit(
            X_train_seq,
            y_train_seq,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        self.is_trained = True
        print("Model training complete.")
        return history

    def evaluate_and_plot(self, X_test_scaled, y_test_scaled, feature_scaler, target_scaler):
        """
        Evaluates the model and plots the results.
        Args:
            X_test_scaled (np.ndarray): Scaled test features.
            y_test_scaled (np.ndarray): Scaled test target.
            feature_scaler (MinMaxScaler): Scaler used for features.
            target_scaler (MinMaxScaler): Scaler used for the target.
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")

        print("\nPreparing test data and making predictions...")
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, self.time_steps)

        # Predictions will be in the scaled format
        y_pred_scaled = self.model.predict(X_test_seq)

        # Inverse transform to get actual prices
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        y_test_actual = target_scaler.inverse_transform(y_test_seq.reshape(-1, 1))

        # --- Evaluation ---
        print("\nEvaluating model performance...")
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
        mae = mean_absolute_error(y_test_actual, y_pred)
        r2 = r2_score(y_test_actual, y_pred)

        print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
        print(f"Mean Absolute Error (MAE):      ${mae:,.2f}")
        print(f"R-squared (R²):                 {r2:.4f}")

        print("\nVisualizing the results...")
        plt.figure(figsize=(15, 7))
        plt.plot(y_test_actual, color='blue', label='Actual Price')
        plt.plot(y_pred, color='red', linestyle='--', label='Predicted Price (LSTM)')
        plt.title('Bitcoin Price Prediction: LSTM Model')
        plt.xlabel('Time (days)')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()



        return {"RMSE": rmse, "MAE": mae, "R2": r2}
