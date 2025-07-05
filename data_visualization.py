
import pandas as pd
import matplotlib.pyplot as plt


def plot_data(df: pd.DataFrame, title_prefix: str = ""):
    """
    Plots the closing price and trading volume of the given DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame containing 'Close' and 'Volume' columns.
        title_prefix (str): A prefix to add to the plot titles (e.g., "Raw Data -").
    """
    if df is None:
        print("DataFrame is None. Cannot plot.")
        return

    # --- Plot 1: Closing Price Only ---
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Close'], label='Closing Price', color='blue')
    plt.title(f'{title_prefix}Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()