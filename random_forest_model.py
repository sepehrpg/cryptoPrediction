import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class RandomForestModel:

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=1.0, random_state=42):

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_trained = False
        print("🚀 Enhanced Random Forest initialized with custom params.")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        print("Training the Random Forest model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Model training complete.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")
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
        # این متد بدون تغییر باقی می‌ماند
        print("\nVisualizing the prediction results...")
        y_pred = self.predict(X_test)
        results_df = pd.DataFrame({
            'Actual Price': y_test,
            'Predicted Price': y_pred
        }, index=y_test.index)
        plt.figure(figsize=(15, 7))
        plt.plot(results_df.index, results_df['Actual Price'], label='Actual Price', color='blue', alpha=0.8)
        plt.plot(results_df.index, results_df['Predicted Price'], label='Predicted Price',
                 color='orange', linestyle='--')
        plt.title('Bitcoin Price Prediction: Actual vs. Predicted')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_feature_importance(self, feature_columns: list):
        # این متد بدون تغییر باقی می‌ماند
        print("\nVisualizing feature importances...")
        if not self.is_trained:
            raise RuntimeError("Model must be trained before plotting.")
        importances = self.model.feature_importances_
        feature_importance_df = pd.Series(importances, index=feature_columns).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        feature_importance_df.plot(kind='bar')
        plt.title('Feature Importance')
        plt.ylabel('Importance Score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()