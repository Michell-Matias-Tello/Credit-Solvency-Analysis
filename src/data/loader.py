import pandas as pd
import os

def load_dataset(filepath: str = 'data/credit_solvency_dataset.csv') -> pd.DataFrame:
    '''Load the dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file. Default: 'data/credit_solvency_dataset.csv'.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
    '''
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} does not exist.")
    return pd.read_csv(filepath)

# Global variables for feature classification
CONTINUOUS_VARS = [
    'age',
    'gross_monthly_income',
    'years_continuous_experience',
    'current_debt_to_income_ratio'
]

DISCRETE_VARS = ['number_of_dependents']
BINARY_VARS = ['home_ownership']
TARGET = 'default_status'

ALL_FEATURES = CONTINUOUS_VARS + DISCRETE_VARS + BINARY_VARS
