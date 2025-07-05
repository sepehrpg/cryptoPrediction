
import pandas as pd
from typing import Tuple
from data_collection_yfinance import get_crypto_data


def split_data_for_time_series(df: pd.DataFrame, target_column: str, train_ratio: float = 0.8) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits data for time series modeling.

    This function splits the data chronologically, which is the correct method for
    time series forecasting. It ensures the model is trained on past data and
    validated on future data, preventing data leakage.

    Args:
        df (pd.DataFrame): The complete DataFrame to be split.
        target_column (str): The name of the target variable column (e.g., 'Close').
        train_ratio (float): The proportion of the dataset to allocate to the training set.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing
        X_train, X_test, y_train, y_test.
    """
    print("\nSplitting data for time series...")
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Calculate the split point index
    split_point = int(len(df) * train_ratio)

    # Split the data chronologically
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]

    print(f"Data split complete: {len(X_train)} rows for training, {len(X_test)} rows for testing.")
    return X_train, X_test, y_train, y_test



if __name__ == "__main__":
    # 1. Get the data
    bitcoin_df = get_crypto_data(ticker="BTC-USD", period="5y")

    # 2. Analyze the data
    if bitcoin_df is not None:
        X_train, X_test, y_train, y_test = split_data_for_time_series(bitcoin_df, target_column="Close", train_ratio=0.8)