import numpy as np
import pandas as pd
import os

# Set random seed for reproducibility
np.random.seed(42)

# Number of rows 
n_rows = 5000


# ============================================================
# GENERATE DATASET
# ============================================================
np.random.seed(42)
n_rows = 5000

print("Generating synthetic credit solvency dataset...")
print(f"Target size: {n_rows} rows\n")

# Generate synthetic data with realistic financial patterns

# Age: working population, centered around 38 with slight right skew
age = np.random.normal(38, 12, n_rows).astype(int)
age = np.clip(age, 20, 75)

# Years of Continuous Work Experience: correlated with age but capped
years_experience = (age - np.random.normal(20, 4, n_rows)).astype(int)
years_experience = np.clip(years_experience, 0, 50)
# Ensure experience doesn't exceed age - 16
mask = years_experience > (age - 16)
years_experience[mask] = (age[mask] - 16)

# Gross Monthly Income: log-normal distribution for realistic income skew
income = np.random.lognormal(mean=8.2, sigma=0.6, size=n_rows)
income = np.round(income, -2)  # Round to nearest hundred
income = np.clip(income, 800, 35000)

# Boost income slightly with experience (realistic pattern)
income = income * (1 + (years_experience / 80))
income = np.round(income, -2)
income = income.astype(int)

# Current Debt-to-Income Ratio (0 to 1): influenced by income
base_dti = np.random.beta(2, 5, n_rows)
income_normalized = (income - income.min()) / (income.max() - income.min())
debt_to_income = base_dti * (1 - income_normalized * 0.4)
debt_to_income = np.round(debt_to_income, 3)
debt_to_income = np.clip(debt_to_income, 0.0, 0.95)

# Number of Dependents: poisson-like distribution
dependents = np.random.poisson(lam=1.2, size=n_rows)
dependents = np.clip(dependents, 0, 6)

# Home Ownership: influenced by income and age
home_ownership_prob = 0.15 + (income_normalized * 0.4) + (np.clip(age, 20, 60) - 20) / 120
home_ownership_prob = np.clip(home_ownership_prob, 0.05, 0.85)
home_ownership = np.random.binomial(1, home_ownership_prob)

# Target Variable: Default Status (1 = Defaulter, 0 = Non-Defaulter)
risk_score = (
    - 0.03 * (age - 20)
    - 0.00002 * (income - 800)
    - 0.02 * years_experience
    + 2.5 * debt_to_income
    + 0.4 * dependents
    - 0.8 * home_ownership
    + np.random.normal(0, 0.8, n_rows)
)

# Convert risk score to probability using sigmoid function
probability_default = 1 / (1 + np.exp(-risk_score))

# Overall default rate targeted around 18%
threshold = np.percentile(probability_default, 82)
default_status = (probability_default >= threshold).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'age': age,
    'gross_monthly_income': income,
    'years_continuous_experience': years_experience,
    'current_debt_to_income_ratio': debt_to_income,
    'number_of_dependents': dependents,
    'home_ownership': home_ownership,
    'default_status': default_status
})

# Display dataset information
print("=" * 60)
print("DATASET GENERATED SUCCESSFULLY")
print("=" * 60)
print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumn names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print(f"\nData types:")
print(df.dtypes.to_string())

print(f"\nFirst 5 rows:")
print(df.head().to_string())

print(f"\nStatistical Summary:")
print(df.describe().round(2).to_string())

print(f"\nClass Distribution (Target Variable):")
count_0 = df['default_status'].value_counts().get(0, 0)
count_1 = df['default_status'].value_counts().get(1, 0)
pct_0 = count_0 / n_rows * 100
pct_1 = count_1 / n_rows * 100
print(f"  Non-Defaulter (0): {count_0} rows ({pct_0:.1f}%)")
print(f"  Defaulter (1):     {count_1} rows ({pct_1:.1f}%)")

print(f"\nMissing values:")
print(df.isnull().sum().to_string())


# Save dataset
dataset_path = os.path.join('data', 'credit_solvency_dataset.csv')
df.to_csv(dataset_path, index=False)

print(f"  ✓ Dataset saved: {dataset_path}")
print(f"  ✓ Shape: {df.shape}")
print(f"  ✓ Default rate: {df['default_status'].mean():.1%}")