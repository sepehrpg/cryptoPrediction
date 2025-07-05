import yfinance as yf
import pandas as pd
from data_visualization import plot_data

def get_crypto_data(ticker: str = "BTC-USD", period: str = "5y") :
    """
    Downloads historical cryptocurrency data from Yahoo Finance and cleans the column index.
    Args:
        ticker (str): The ticker symbol to download (e.g., "BTC-USD").
        period (str): The time period for which to download data (e.g., "5y", "1mo", "max").

    Returns:
        Optional[pd.DataFrame]: A DataFrame with a single-level column index containing
                                the historical data, or None if the download fails.
    """
    try:
        print(f"Downloading {period} of data for {ticker}...")
        df = yf.download(ticker, period=period, auto_adjust=True)

        if df.empty:
            print(f"No data found for ticker {ticker}. Please check the symbol.")
            return None

        # BEST PRACTICE: Fix potential MultiIndex columns at the source.
        # yfinance can return columns as a MultiIndex. This flattens it to a standard index.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        print("Data downloaded successfully.")

        # Tip : index is datetimeIndex
        # Ensure only the essential columns are present and in the correct order
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        return df
    except Exception as e:
        print(f"An error occurred during download: {e}")
        return None


# This block of code will only run when you execute the file directly
# It will NOT run when this file is imported into another script.
if __name__ == "__main__":
    # 1. Get the data
    bitcoin_df = get_crypto_data(ticker="BTC-USD", period="5y")

    # 2. Analyze the data
    if bitcoin_df is not None:
        pass
        plot_data(bitcoin_df)
