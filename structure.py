# ============================================
# SETUP_PROJECT.PY
# Script to create the folder and base file structure
# for a Python data analysis project.
# Run this ONCE at the start of the project.
# ============================================

import os

# ============================================
# 1. DEFINE FOLDER STRUCTURE
# ============================================
FOLDER_STRUCTURE = {
    # Main folders
    "":
    [
        "data",               # Raw and processed data
        "notebooks",         # Jupyter Notebooks (as .py or .ipynb)
        "src",                # Modular source code
        "outputs/figures",    # Generated plots
        "models",             # Trained models
        "tests",              # Unit tests
        "docs",               # Project documentation
    ],
    # Subfolders inside 'src'
    "src":
    [
        "data",              # Data loading and preprocessing module
        "models",            # Machine learning models module
        "visualization",     # Visualization module
    ],
}

# ============================================
# 2. CREATE FOLDERS
# ============================================
print("Creating project folder structure...")
for root, subfolders in FOLDER_STRUCTURE.items():
    for folder in subfolders:
        folder_path = os.path.join(root, folder) if root else folder
        os.makedirs(folder_path, exist_ok=True)
        print(f"  ✅ Folder created: {folder_path}/")

# ============================================
# 3. CREATE BASE FILES IN 'src/'
# ============================================
print("\nCreating base files in 'src/'...")

# --- 3.1. __init__.py files ---
init_files = [
    "src/__init__.py",
    "src/data/__init__.py",
    "src/models/__init__.py",
    "src/visualization/__init__.py",
]

for init_file in init_files:
    with open(init_file, "w", encoding='utf-8') as f:
        f.write("# Python package\n")
    print(f"  ✅ File created: {init_file}")

# --- 3.2. src/data/loader.py ---
with open("src/data/loader.py", "w", encoding='utf-8') as f:
    f.write("""import pandas as pd
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
""")
print("  ✅ File created: src/data/loader.py")

# --- 3.3. src/data/preprocessing.py ---
with open("src/data/preprocessing.py", "w", encoding='utf-8') as f:
    f.write("""import pandas as pd
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
    print(f"Dimensions: {df.shape[0]:,} rows x {df.shape[1]} columns\\n")
    print("Column names and data types:")
    print(df.dtypes.to_string())
    print(f"\\nTotal missing values: {df.isnull().sum().sum()}")
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
""")
print("  ✅ File created: src/data/preprocessing.py")

# --- 3.4. src/models/train.py ---
with open("src/models/train.py", "w", encoding='utf-8') as f:
    f.write("""import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple

def calculate_risk_score(
    df: pd.DataFrame,
    seed: int = 42
) -> Tuple[pd.Series, float]:
    '''Calculate risk score and default probability for each record.

    Args:
        df (pd.DataFrame): Dataset with columns: age, gross_monthly_income,
                           years_continuous_experience, current_debt_to_income_ratio,
                           number_of_dependents, home_ownership.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.Series, float]: (default_status, default_rate).
    '''
    np.random.seed(seed)
    n_rows = len(df)

    age = df['age'].values
    income = df['gross_monthly_income'].values
    years_experience = df['years_continuous_experience'].values
    debt_to_income = df['current_debt_to_income_ratio'].values
    dependents = df['number_of_dependents'].values
    home_ownership = df['home_ownership'].values

    # Risk score calculation
    risk_score = (
        -0.03 * (age - 20)
        - 0.00002 * (income - 800)
        - 0.02 * years_experience
        + 2.5 * debt_to_income
        + 0.4 * dependents
        - 0.8 * home_ownership
        + np.random.normal(0, 0.8, n_rows)
    )

    # Convert to probability using sigmoid function
    probability_default = 1 / (1 + np.exp(-risk_score))

    # Threshold for 18% default rate
    threshold = np.percentile(probability_default, 82)
    default_status = (probability_default >= threshold).astype(int)
    default_rate = default_status.mean()

    return default_status, default_rate
""")
print("  ✅ File created: src/models/train.py")

# --- 3.5. src/visualization/plots.py ---
with open("src/visualization/plots.py", "w", encoding='utf-8') as f:
    f.write("""import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

# Global visualization style
def set_visualization_style():
    '''Set the global visualization style for the project.'''
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("muted")
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['axes.labelsize'] = 11

# Color palette
COLOR_SOLVENT = '#2ecc71'
COLOR_DEFAULTER = '#e74c3c'
COLOR_NEUTRAL = '#3498db'
PALETTE_DICT = {0: COLOR_SOLVENT, 1: COLOR_DEFAULTER}

def plot_target_distribution(df: pd.DataFrame, target: str = 'default_status') -> None:
    '''Plot the distribution of the target variable.

    Args:
        df (pd.DataFrame): Dataset with the target column.
        target (str): Name of the target column.
    '''
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=target, palette=PALETTE_DICT)
    plt.title(f"Target Variable Distribution ({target})")
    plt.xlabel("Status")
    plt.ylabel("Frequency")
    plt.show()
""")
print("  ✅ File created: src/visualization/plots.py")

# --- 3.6. src/utils.py ---
with open("src/utils.py", "w", encoding='utf-8') as f:
    f.write("""import pandas as pd

def print_dataset_summary(df: pd.DataFrame) -> None:
    '''Print a comprehensive summary of the dataset.

    Args:
        df (pd.DataFrame): Dataset to summarize.
    '''
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"\\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"\\nColumn names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")

    print(f"\\nData types:")
    print(df.dtypes.to_string())

    print(f"\\nFirst 5 rows:")
    print(df.head().to_string())

    print(f"\\nStatistical summary:")
    print(df.describe().round(2).to_string())

    if 'default_status' in df.columns:
        print(f"\\nTarget Variable Distribution (default_status):")
        count_0 = df['default_status'].value_counts().get(0, 0)
        count_1 = df['default_status'].value_counts().get(1, 0)
        pct_0 = count_0 / len(df) * 100
        pct_1 = count_1 / len(df) * 100
        print(f"  Non-Defaulter (0): {count_0:,} rows ({pct_0:.1f}%)")
        print(f"  Defaulter (1):     {count_1:,} rows ({pct_1:.1f}%)")

    print(f"\\nMissing values per column:")
    print(df.isnull().sum().to_string())
""")
print("  ✅ File created: src/utils.py")

# ============================================
# 4. CREATE BASE FILES IN 'tests/' AND 'docs/'
# ============================================
print("\nCreating base files in 'tests/' and 'docs/'...")

# --- 4.1. Base files for unit tests in 'tests/' ---
with open("tests/__init__.py", "w", encoding='utf-8') as f:
    f.write("# __init__.py for tests directory\n")

with open("tests/test_data_loader.py", "w", encoding='utf-8') as f:
    f.write("""import unittest
from src.data.loader import load_dataset

class TestDataLoader(unittest.TestCase):
    def test_load_dataset(self):
        data = load_dataset('data/credit_solvency_dataset.csv')
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)

if __name__ == '__main__':
    unittest.main()
""")
print("  ✅ Files created in 'tests/'")

# --- 4.2. Base files for documentation in 'docs/' ---
with open("docs/__init__.py", "w", encoding='utf-8') as f:
    f.write("# __init__.py for docs directory\n")

with open("docs/project_overview.md", "w", encoding='utf-8') as f:
    f.write("""# Credit Solvency Project Overview

## Description
This project analyzes credit solvency using synthetic data.

## Structure
- `data/`: Contains the generated dataset.
- `src/`: Modular source code.
- `notebooks/`: Exploratory analysis and modeling.
- `outputs/`: Results and trained models.
- `tests/`: Unit tests.
- `docs/`: Project documentation.
- `models/`: Trained models.

## Author
MICHELL MATIAS TELLO
""")
print("  ✅ Files created in 'docs/'")

# ============================================
# 5. CREATE BASE FILES IN 'notebooks/'
# ============================================
print("\nCreating base files in 'notebooks/'...")

# --- 5.1. notebooks/01_data_exploration.py ---
with open("notebooks/01_data_exploration.py", "w", encoding='utf-8') as f:
    f.write("""# Data Exploration
import sys
import os
sys.path.append(os.path.abspath('..'))

from src.data.loader import load_dataset, CONTINUOUS_VARS, DISCRETE_VARS, BINARY_VARS, TARGET
from src.data.preprocessing import inspect_dataset
from src.visualization.plots import set_visualization_style

# Set visualization style
set_visualization_style()

# Load dataset
df = load_dataset()

# Inspect dataset
inspect_dataset(df)

# Feature classification
print("=" * 60)
print("FEATURE CLASSIFICATION")
print("=" * 60)
print(f"Continuous variables:  {CONTINUOUS_VARS}")
print(f"Discrete variables:   {DISCRETE_VARS}")
print(f"Binary variables:     {BINARY_VARS}")
print(f"Target variable:     {TARGET}")
""")
print("  ✅ File created: notebooks/01_data_exploration.py")

# --- 5.2. notebooks/02_data_cleaning.py ---
with open("notebooks/02_data_cleaning.py", "w", encoding='utf-8') as f:
    f.write("""# Data Cleaning and Preprocessing
import sys
import os
sys.path.append(os.path.abspath('..'))

from src.data.loader import load_dataset
from src.data.preprocessing import inspect_dataset, handle_missing_values

# Load dataset
df = load_dataset()

# Inspect missing values
print("\\nMissing values per column:")
print(df.isnull().sum())

# Handle missing values
df_clean = handle_missing_values(df, strategy='drop')
print(f"\\nCleaned dataset: {df_clean.shape[0]:,} rows x {df_clean.shape[1]} columns")
""")
print("  ✅ File created: notebooks/02_data_cleaning.py")

# --- 5.3. notebooks/03_model_training.py ---
with open("notebooks/03_model_training.py", "w", encoding='utf-8') as f:
    f.write("""# Model Training
import sys
import os
sys.path.append(os.path.abspath('..'))

from src.data.loader import load_dataset
from src.models.train import calculate_risk_score
from src.utils import print_dataset_summary

# Load dataset
df = load_dataset()

# Calculate risk score
df['default_status'], default_rate = calculate_risk_score(df)
print(f"\\nGenerated default rate: {default_rate:.1%}")

# Save dataset with risk score
df.to_csv('data/credit_solvency_dataset_with_risk.csv', index=False)
print("Dataset with risk score saved to 'data/credit_solvency_dataset_with_risk.csv'")

# Dataset summary
print_dataset_summary(df)
""")
print("  ✅ File created: notebooks/03_model_training.py")

# --- 5.4. notebooks/04_results_analysis.py ---
with open("notebooks/04_results_analysis.py", "w", encoding='utf-8') as f:
    f.write("""# Results Analysis
import sys
import os
sys.path.append(os.path.abspath('..'))

from src.data.loader import load_dataset
from src.visualization.plots import set_visualization_style, plot_target_distribution

# Set visualization style
set_visualization_style()

# Load dataset with risk score
df = load_dataset('data/credit_solvency_dataset_with_risk.csv')

# Plot target distribution
plot_target_distribution(df)
""")
print("  ✅ File created: notebooks/04_results_analysis.py")

# --- 5.5. notebooks/README.md ---
with open("notebooks/README.md", "w", encoding='utf-8') as f:
    f.write("""# Credit Solvency Analysis Notebooks

This folder contains Jupyter Notebooks for the credit solvency analysis project.

## Contents
- `01_data_exploration.py`: Exploratory data analysis.
- `02_data_cleaning.py`: Data cleaning and preprocessing.
- `03_model_training.py`: Risk model training.
- `04_results_analysis.py`: Results analysis and visualization.

## Requirements
- Python 3.8+
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`
""")
print("  ✅ File created: notebooks/README.md")

# ============================================
# 6. CREATE ADDITIONAL FILES IN ROOT
# ============================================
print("\nCreating additional files in the project root...")

# --- 6.1. .gitignore ---
with open(".gitignore", "w", encoding='utf-8') as f:
    f.write("""# Python temporary files
__pycache__/
*.py[cod]
*$py.class
*.pyo
*.pyd

# Virtual environment
venv/
env/
.venv/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Local data
# data/

# System files
.DS_Store
Thumbs.db

# Output files
/outputs/
""")
print("  ✅ File created: .gitignore")


with open("README.md", "w", encoding='utf-8') as f:
    f.write("""# Credit Solvency Analysis Project

## Description
This project analyzes credit solvency using synthetic data generated with Python.

## Project Structure

credit-solvency-assessment/
│
├── data/
│   └── credit_solvency_dataset.csv
│
├── notebooks/
│   ├── 01_data_exploration.py
│   ├── 02_data_cleaning.py
│   ├── 03_model_training.py
│   ├── 04_results_analysis.py
│   └── README.md
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessing.py
│   ├── models/
│   │   └── train.py
│   ├── visualization/
│   │   └── plots.py
│   └── utils.py
│
├── outputs/
│   └── figures/
│       ├── bivariate_analysis.png
│       ├── categorical_target_dashboard.png
│       ├── confusion_matrices.png
│       ├── correlation_matrix.png
│       ├── density_by_class.png
│       ├── information_value.png
│       ├── model_comparison_metrics.png
│       ├── precision_recall_curves.png
│       ├── risk_bands.png
│       ├── roc_curves_comparison.png
│       ├── score_distribution.png
│       ├── threshold_optimization.png
│       └── univariate_continuous.png
│
├── models/
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   └── xgboost_model.pkl
│
├── tests/
│   ├── __init__.py
│   └── test_data_loader.py
│
├── docs/
│   ├── __init__.py
│   └── project_overview.md
│
├── main.py
├── structure.py
├── data.py
├── .gitignore
├── README.md
└── requirements.txt

## Author
MICHELL MATIAS TELLO
""")
    
print("  ✅ File created: README.md")


