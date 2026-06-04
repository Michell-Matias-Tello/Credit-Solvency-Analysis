import pandas as pd
import numpy as np
from typing import Optional

def inspect_dataset(df: pd.DataFrame) -> pd.DataFrame:
    '''Inspect the dataset structure and display basic info.

    Args:
        df (pd.DataFrame): Dataset to inspect.

    Returns:
        pd.DataFrame: Original dataset.
    '''
    print("=" * 60)
    print("DATASET STRUCTURAL INSPECTION")
    print("=" * 60)
    print(f"Dimensions: {df.shape[0]:,} rows x {df.shape[1]} columns\n")
    print("Column names and data types:")
    print(df.dtypes.to_string())
    print(f"\nTotal missing values: {df.isnull().sum().sum()}")
    return df

def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    '''Handle missing values in the dataset.

    Args:
        df (pd.DataFrame): Dataset to process.
        strategy (str): Strategy to handle missing values ('drop', 'fill_mean', 'fill_median').

    Returns:
        pd.DataFrame: Dataset without missing values.
    '''
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill_mean':
        return df.fillna(df.mean())
    elif strategy == 'fill_median':
        return df.fillna(df.median())
    else:
        raise ValueError(f"Invalid strategy: {strategy}. Use 'drop', 'fill_mean', or 'fill_median'.")
