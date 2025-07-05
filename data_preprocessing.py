
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional
from data_collection_yfinance import get_crypto_data
from data_visualization import plot_data
from data_analyze import data_analyze




def data_preprocessing(
    df: pd.DataFrame,
    add_datetime_features_flag: bool = True,
    create_lagged_features_flag: bool = True,
    lag_days: int = 1,
    add_technical_flag: bool = True,
    scale_data_flag: bool = False,
) -> Tuple[Optional[pd.DataFrame], Optional[MinMaxScaler]]:
    """
    A complete preprocessing pipeline for time series prediction with:
    - Lagged features
    - Technical indicators
    - Date-time features
    - Scaling

    Args:
        df (pd.DataFrame): Raw OHLCV data.
        add_datetime_features_flag (bool): If True, adds Year, Month, Day, DayOfWeek.
        create_lagged_features_flag (bool): If True, creates lagged features.
        lag_days (int): Number of lag days to create (e.g., 1, 3, 5).
        add_technical_flag (bool): Whether to add technical indicators based on lagged Close.
        scale_data_flag (bool): Whether to scale features using MinMaxScaler.

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[MinMaxScaler]]
    """
    if df is None:
        return None, None

    print("📊 Starting data preprocessing pipeline (v2)...")
    processed_df = df.copy()

    # --- Step 1: Create Lagged Features ---
    if create_lagged_features_flag:
        print(f"🔁 Creating lagged features for {lag_days} days...")
        feature_columns_to_lag = ['Open', 'High', 'Low', 'Close', 'Volume']

        for lag in range(1, lag_days + 1):
            for col in feature_columns_to_lag:
                processed_df[f'{col}_lag{lag}'] = processed_df[col].shift(lag)

        # Drop current-day original features to avoid leakage (except target 'Close')
        processed_df.drop(columns=['Open', 'High', 'Low', 'Volume'], inplace=True)

    # --- Step 2: Add Date-Time Features ---
    if add_datetime_features_flag:
        processed_df = add_datetime_features(processed_df)  # Already implemented in your code

    # --- Step 3: Add Technical Indicators ---
    if add_technical_flag:
        if f'Close_lag1' not in processed_df.columns:
            print("⚠️ Technical indicators require at least lag=1 on 'Close'. Adding it now...")
            processed_df['Close_lag1'] = df['Close'].shift(1)

        processed_df = add_technical_indicators(processed_df)

    # --- Step 4: Drop NaN values ---
    processed_df.dropna(inplace=True)

    # --- Step 5: Scale Feature Columns (optional) ---
    scaler = None
    if scale_data_flag:
        print("⚖️ Scaling data...")
        target_column = 'Close'

        if target_column in processed_df.columns:
            feature_columns_to_scale = processed_df.columns.drop(target_column)
            scaler = MinMaxScaler()
            processed_df[feature_columns_to_scale] = scaler.fit_transform(processed_df[feature_columns_to_scale])
            print("✅ Data scaled successfully.")
        else:
            print(f"⚠️ Target column '{target_column}' not found. Skipping scaling.")

    print("✅ Preprocessing complete.")
    return processed_df, scaler




def data_preprocessing_old(
        df: pd.DataFrame,
        add_date_time_features: bool = False,
        scale_data_flag: bool = True,
        add_technical_flag: bool = False,
) -> Tuple[Optional[pd.DataFrame], Optional[MinMaxScaler]]:
    """
    The main pipeline to run the entire process of data preprocessing.
    Args:
        df (pd.DataFrame): Raw DataFrame from yfinance.
        add_date_time_features (bool): If True, adds datetime features. (year,month,day,dayOfWeak)
        scale_data_flag (bool): If True, scales the numerical data. (normalize)
        add_technical_flag (bool):

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[MinMaxScaler]]: A tuple containing the
        processed DataFrame and the fitted scaler instance (or None if not scaled).
    """
    if df is None:
        return None, None

    # Create a copy to ensure the original DataFrame remains untouched
    processed_df = df.copy()

    # Step 2: Add Date Time Feature
    if add_date_time_features:
        processed_df = add_datetime_features(processed_df)

    # --- Step 3: Add Technical Indicators ---
    if add_technical_flag:
        processed_df = add_technical_indicators(processed_df)

    # Step 4: Scale Data
    scaler = None
    if scale_data_flag:
        columns_to_scale = ["Open", "High", "Low", "Close", "Volume"]
        processed_df, scaler = scale_data(processed_df, columns_to_scale)

    return processed_df, scaler


def data_preprocessing_shifted(
        df: pd.DataFrame,
        add_datetime_features_flag: bool = True,
        create_lagged_features_flag: bool = True,
        scale_data_flag: bool = True
) -> Tuple[Optional[pd.DataFrame], Optional[MinMaxScaler]]:
    """
    A comprehensive pipeline that prepares raw time series data for modeling.
    It creates lagged features to prevent data leakage and optionally scales the data.

    Args:
        df (pd.DataFrame): Raw DataFrame from yfinance (OHLCV).
        add_datetime_features_flag (bool): If True, adds datetime features.
        create_lagged_features_flag (bool): If True, creates lagged features to prevent data leakage.
        scale_data_flag (bool): If True, scales the feature data.


    Returns:
        Tuple[Optional[pd.DataFrame], Optional[MinMaxScaler]]: A tuple containing the
        processed DataFrame and the fitted scaler instance (or None if not scaled).
    """
    if df is None:
        return None, None

    print("Starting data preprocessing pipeline...")
    processed_df = df.copy()

    # --- Step 1: Create Lagged Features (Solves Data Leakage) ---
    if create_lagged_features_flag:
        print("Creating lagged features...")
        # The target column ('Close') is the price of the CURRENT day. We keep it.
        # The features will be the data from the PREVIOUS day.

        feature_columns_to_lag = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in feature_columns_to_lag:
            # Create new columns with a '_lag1' suffix for previous day's data
            processed_df[f'{col}_lag1'] = processed_df[col].shift(1)

        # Drop original feature columns that would cause data leakage, but KEEP the target 'Close'.
        columns_to_drop = ['Open', 'High', 'Low', 'Volume']
        processed_df.drop(columns=columns_to_drop, inplace=True)

    # --- Step 2: Add Datetime Features ---
    if add_datetime_features_flag:
        processed_df = add_datetime_features(processed_df)



    # --- Step 3: Handle NaN Values created by .shift() ---
    print("Dropping rows with NaN values...")
    processed_df.dropna(inplace=True)

    # --- Step 4: Scale Data if requested ---
    scaler = None
    if scale_data_flag:
        print("Scaling feature data...")
        # The target column is 'Close'. All other columns are features to be scaled.
        target_column = 'Close'

        # Make sure the target column exists before trying to drop it from the feature list
        if target_column in processed_df.columns:
            feature_columns_to_scale = processed_df.columns.drop(target_column)

            # Create a copy to avoid SettingWithCopyWarning
            df_to_scale = processed_df.copy()

            scaler = MinMaxScaler()
            # Fit and transform only the feature columns
            df_to_scale[feature_columns_to_scale] = scaler.fit_transform(df_to_scale[feature_columns_to_scale])
            processed_df = df_to_scale
            print("Feature data successfully scaled.")
        else:
            # This case happens if create_lagged_features_flag was False and the original 'Close' was dropped.
            # The logic is now corrected, but this is a safeguard.
            print(f"Warning: Target column '{target_column}' not found. Skipping scaling.")

    print("Data preprocessing pipeline complete.")
    return processed_df, scaler


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds time-based features (Year, Month, Day, DayOfWeek) to the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame with a DatetimeIndex.

    Returns:
        pd.DataFrame: The DataFrame with added datetime features.
    """
    print("\nAdding datetime features...")
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['DayOfWeek'] = df.index.dayofweek
    print("datetime features Added")
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators to the DataFrame.
    These are calculated based on lagged data to prevent data leakage.
    """
    print("Adding technical indicators...")

    # Simple Moving Averages (SMA)
    # Helps the model understand the short-term and long-term trends.
    df['SMA_7'] = df['Close_lag1'].rolling(window=7).mean()
    df['SMA_21'] = df['Close_lag1'].rolling(window=21).mean()

    # Relative Strength Index (RSI)
    # Helps the model understand momentum and overbought/oversold conditions.
    delta = df['Close_lag1'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    # Helps the model understand market volatility.
    df['BB_Middle'] = df['Close_lag1'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close_lag1'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close_lag1'].rolling(window=20).std()

    print("Technical indicators added.")
    return df


def scale_data(df: pd.DataFrame, columns_to_scale: List[str]) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Scales numerical data to the [0, 1] range using MinMaxScaler.
    Args:
        df (pd.DataFrame): The input DataFrame.
        columns_to_scale (List[str]): A list of column names to be scaled.

    Returns:
        Tuple[pd.DataFrame, MinMaxScaler]: A tuple containing the scaled DataFrame
                                           and the fitted scaler instance for later use.
    """
    print("\nScaling data...")
    feature_scaler = MinMaxScaler()
    df[columns_to_scale] = feature_scaler.fit_transform(df[columns_to_scale])
    print("Data successfully scaled.")
    return df, feature_scaler


def create_lagged_features_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates lagged features to prevent data leakage.
    Uses yesterday's data to predict today's price.
    """
    if df is None:
        return None

    df_lagged = df.copy()

    # --- Step 1: Create the target variable BEFORE shifting ---
    # The target is today's actual closing price
    df_lagged['target'] = df_lagged['Close']

    # --- Step 2: Create lagged features ---
    # These are the features from the previous day (t-1)
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in feature_columns:
        df_lagged[f'{col}_lag1'] = df_lagged[col].shift(1)

    # --- Step 3: Add time features ---
    # These features do not need to be lagged
    df_lagged = add_datetime_features(df_lagged)  # Assuming you have this function

    # --- Step 4: Drop rows with NaN values ---
    # The first row will have NaN for lagged features, so we drop it
    df_lagged.dropna(inplace=True)

    # --- Step 5: Select final features and target ---
    # The target is the non-lagged 'target' column we created
    y = df_lagged['target']

    # The features are the lagged columns and the time features
    # IMPORTANT: We drop the original, non-lagged price/volume columns
    # and the new 'target' column from our feature set X
    X = df_lagged.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'target'])

    print("Lagged features created successfully.")
    return X, y


# This block ensures that the code inside it only runs when the script is executed directly
if __name__ == "__main__":
    dc = get_crypto_data()
    plot_data(dc)
    data_analyze(dc)
    df,scaled = data_preprocessing(dc)
    plot_data(df)
    data_analyze(df)
