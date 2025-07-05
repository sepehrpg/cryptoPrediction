
import pandas as pd
from data_collection_yfinance import get_crypto_data


def data_analyze(df: pd.DataFrame, source_name: str = ""):
    """
    Performs and prints a basic analysis of the DataFrame, including description
    and missing values check.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.
        source_name (str): An identifier for the data source (e.g., "Raw", "Scaled").
    """
    if df is None:
        print("DataFrame is empty. Skipping analysis.")
        return

    print(f"\n===== Analysis of {source_name} Data =====")

    # Display the first few rows of the data
    print("\nFirst few rows of the dataset:")
    print(df.head())

    # Display summary statistics
    print("\nSummary statistics:")
    print(df.describe())

    # Check for missing values
    print("\nNumber of missing values in each column:")
    missing_data = df.isnull().sum()
    print(missing_data)

    # Display dataset information
    print("\nDataset information:")
    df.info()


if __name__ == "__main__":
    bitcoin_df = get_crypto_data(ticker="BTC-USD", period="5y")

    # 2. Analyze the data
    if bitcoin_df is not None:
        data_analyze(bitcoin_df)
