# Results Analysis
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
