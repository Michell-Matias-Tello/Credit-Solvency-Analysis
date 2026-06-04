# Credit Solvency Analysis Project

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
