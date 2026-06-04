# Data Exploration
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
