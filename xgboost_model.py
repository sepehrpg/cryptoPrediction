import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class XGBoostModel:
    def __init__(self,
                 n_estimators=100,
                 learning_rate=0.1,
                 max_depth=6,
                 random_state=42,
                 **kwargs):
        """
        Initialize the XGBoost regressor model.

        Args:
            n_estimators (int): Number of boosting rounds.
            learning_rate (float): Step size shrinkage.
            max_depth (int): Maximum tree depth.
            random_state (int): Random seed.
            kwargs: Other XGBRegressor parameters.
        """
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            **kwargs
        )
        self.is_trained = False
        print("XGBoost model initialized.")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        print("Training XGBoost model... (this may take some time)")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Training complete.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet. Please call train() first.")
        return self.model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        print("\nEvaluating model performance...")
        y_pred = self.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
        print(f"Mean Absolute Error (MAE):      ${mae:,.2f}")
        print(f"R-squared (R²):                 {r2:.4f}")

        return {"RMSE": rmse, "MAE": mae, "R2": r2}

    def plot_results(self, X_test: pd.DataFrame, y_test: pd.Series):
        print("\nPlotting actual vs predicted values...")
        y_pred = self.predict(X_test)

        df_results = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        }, index=y_test.index)

        plt.figure(figsize=(14, 7))
        plt.plot(df_results.index, df_results['Actual'], label='Actual Price', color='blue', alpha=0.8)
        plt.plot(df_results.index, df_results['Predicted'], label='Predicted Price (XGBoost)', color='orange', linestyle='--')
        plt.title('Bitcoin Price Prediction: Actual vs Predicted (XGBoost)')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_feature_importance(self, feature_columns: list):
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet. Train before plotting feature importance.")

        print("\nPlotting feature importances...")
        importances = self.model.feature_importances_

        importance_df = pd.Series(importances, index=feature_columns).sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        importance_df.plot(kind='bar')
        plt.title('Feature Importance (XGBoost)')
        plt.ylabel('Importance Score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
