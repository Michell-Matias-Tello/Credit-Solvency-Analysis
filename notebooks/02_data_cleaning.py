# Data Cleaning and Preprocessing
import sys
import os
sys.path.append(os.path.abspath('..'))

from src.data.loader import load_dataset
from src.data.preprocessing import inspect_dataset, handle_missing_values

# Load dataset
df = load_dataset()

# Inspect missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Handle missing values
df_clean = handle_missing_values(df, strategy='drop')
print(f"\nCleaned dataset: {df_clean.shape[0]:,} rows x {df_clean.shape[1]} columns")
