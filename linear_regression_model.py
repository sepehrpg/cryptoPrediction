from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from typing import Optional


class LinearRegressionModel:
    """
    A flexible class for the Linear Regression model that can handle both
    scaled and unscaled target variables by checking if a scaler is provided.
    """

    def __init__(self, scaler: Optional[MinMaxScaler] = None):
        """
        Initializes the LinearRegressionModel.

        Args:
            scaler (Optional[MinMaxScaler]): The scaler object fitted on the data.
                                              If None, predictions and evaluations are
                                              assumed to be on the original scale.
        """
        self.model = LinearRegression()
        self.scaler = scaler
        self.is_trained = False
        print("Linear Regression model initialized (Flexible Version).")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Trains the Linear Regression model.

        Args:
            X_train (pd.DataFrame): The training features (can be scaled or unscaled).
            y_train (pd.Series): The training target values (can be scaled or unscaled).
        """
        print("Training the model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Model training complete.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions on the input feature data.

        Args:
            X (pd.DataFrame): The input features for prediction.

        Returns:
            np.ndarray: The predictions. The scale depends on what the model was trained on.
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet. Please call the 'train' method first.")

        return self.model.predict(X)

    def _get_unscaled_results(self, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """
        A helper method to get predictions and ensure both predictions and actual values
        are returned in their original, unscaled form for correct evaluation.
        """
        y_pred = self.predict(X_test)

        if self.scaler:
            # If a scaler is provided, it means y_test and y_pred are scaled.
            # We must inverse_transform them to get meaningful metrics.
            print("Scaler found. Inverse transforming results for evaluation...")

            # This assumes the scaler was fitted on a DataFrame where 'Close' was at index 3.
            # This is based on the previous context: columns_to_scale = ["Open", "High", "Low", "Close", "Volume"]
            target_column_index = 3
            num_features = self.scaler.n_features_in_

            # Unscale predictions
            temp_pred = np.zeros((len(y_pred), num_features))
            temp_pred[:, target_column_index] = y_pred
            y_pred_unscaled = self.scaler.inverse_transform(temp_pred)[:, target_column_index]

            # Unscale actual test values
            temp_y = np.zeros((len(y_test), num_features))
            temp_y[:, target_column_index] = y_test.values
            y_test_unscaled = self.scaler.inverse_transform(temp_y)[:, target_column_index]

            return y_test_unscaled, y_pred_unscaled
        else:
            # If no scaler, it means y_test and y_pred are already in the original scale.
            print("No scaler provided. Evaluating on original scale.")
            return y_test.values, y_pred

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Evaluates the model on the test set and prints the metrics.
        It correctly handles both scaled and unscaled scenarios.
        """
        print("\nEvaluating model performance...")
        y_true_final, y_pred_final = self._get_unscaled_results(X_test, y_test)

        # Calculate metrics on the unscaled (real-world) values
        rmse = np.sqrt(mean_squared_error(y_true_final, y_pred_final))
        mae = mean_absolute_error(y_true_final, y_pred_final)
        r2 = r2_score(y_true_final, y_pred_final)

        print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
        print(f"Mean Absolute Error (MAE):      ${mae:,.2f}")
        print(f"R-squared (R²):                 {r2:.4f}")

        return {"RMSE": rmse, "MAE": mae, "R2": r2}

    def plot_results(self, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Plots the actual vs. predicted prices, ensuring values are on the original scale.
        """
        print("\nVisualizing the results...")
        y_test_unscaled, y_pred_unscaled = self._get_unscaled_results(X_test, y_test)

        # Create a DataFrame for easy plotting
        results_df = pd.DataFrame({
            'Actual Price': y_test_unscaled,
            'Predicted Price': y_pred_unscaled
        }, index=y_test.index)

        plt.figure(figsize=(15, 7))
        plt.plot(results_df.index, results_df['Actual Price'], label='Actual Price', color='blue')
        plt.plot(results_df.index, results_df['Predicted Price'], label='Predicted Price', color='red', linestyle='--')
        plt.title('Bitcoin Price Prediction: Actual vs. Predicted')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()


