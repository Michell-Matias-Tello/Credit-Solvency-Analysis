import matplotlib.pyplot as plt
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
