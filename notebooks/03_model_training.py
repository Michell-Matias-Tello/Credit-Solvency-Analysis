# Model Training
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
print(f"\nGenerated default rate: {default_rate:.1%}")

# Save dataset with risk score
df.to_csv('data/credit_solvency_dataset_with_risk.csv', index=False)
print("Dataset with risk score saved to 'data/credit_solvency_dataset_with_risk.csv'")

# Dataset summary
print_dataset_summary(df)
