import numpy as np
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
