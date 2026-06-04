import pandas as pd

def print_dataset_summary(df: pd.DataFrame) -> None:
    '''Print a comprehensive summary of the dataset.

    Args:
        df (pd.DataFrame): Dataset to summarize.
    '''
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"\nColumn names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")

    print(f"\nData types:")
    print(df.dtypes.to_string())

    print(f"\nFirst 5 rows:")
    print(df.head().to_string())

    print(f"\nStatistical summary:")
    print(df.describe().round(2).to_string())

    if 'default_status' in df.columns:
        print(f"\nTarget Variable Distribution (default_status):")
        count_0 = df['default_status'].value_counts().get(0, 0)
        count_1 = df['default_status'].value_counts().get(1, 0)
        pct_0 = count_0 / len(df) * 100
        pct_1 = count_1 / len(df) * 100
        print(f"  Non-Defaulter (0): {count_0:,} rows ({pct_0:.1f}%)")
        print(f"  Defaulter (1):     {count_1:,} rows ({pct_1:.1f}%)")

    print(f"\nMissing values per column:")
    print(df.isnull().sum().to_string())
