# Credit Solvency Assessment

## 📌 Project Overview

This project focuses on **assessing credit solvency** through advanced data analysis and machine learning techniques. The goal is to develop robust models capable of predicting creditworthiness based on historical and behavioral data. The project leverages unsupervised and supervised learning methods to identify patterns, segment users, and generate actionable insights for risk assessment.

The repository is structured to ensure **modularity, reproducibility, and scalability**, making it suitable for both research and production environments.

---

## 🗂️ Repository Structure

```
credit-solvency-assessment/
│
├── data/
│   └── credit_solvency_dataset.csv          # Raw dataset for analysis
│
├── notebooks/                                # Jupyter notebooks for exploratory analysis
│   ├── 01_data_exploration.py               # Initial data exploration and visualization
│   ├── 02_data_cleaning.py                  # Data preprocessing and cleaning
│   ├── 03_model_training.py                 # Model training and validation
│   ├── 04_results_analysis.py               # Performance evaluation and insights
│   └── README.md                             # Notebooks usage guidelines
│
├── src/                                      # Source code for reusable components
│   ├── data/
│   │   ├── loader.py                         # Data loading utilities
│   │   └── preprocessing.py                  # Data preprocessing functions
│   ├── models/
│   │   └── train.py                          # Model training logic
│   ├── visualization/
│   │   └── plots.py                          # Custom visualization functions
│   └── utils.py                              # Utility functions and helpers
│
├── outputs/                                  # Generated outputs and visualizations
│   └── figures/
│       ├── bivariate_analysis.png           # Bivariate relationships
│       ├── categorical_target_dashboard.png # Categorical feature analysis
│       ├── confusion_matrices.png           # Model confusion matrices
│       ├── correlation_matrix.png            # Feature correlation heatmap
│       ├── density_by_class.png              # Density plots by target class
│       ├── information_value.png             # Information value analysis
│       ├── model_comparison_metrics.png      # Model performance comparison
│       ├── precision_recall_curves.png       # Precision-recall curves
│       ├── risk_bands.png                    # Risk segmentation visualization
│       ├── roc_curves_comparison.png         # ROC curves for model comparison
│       ├── score_distribution.png            # Score distribution analysis
│       ├── threshold_optimization.png        # Threshold tuning results
│       └── univariate_continuous.png          # Univariate analysis of continuous features
│
├── models/                                   # Serialized trained models
│   ├── logistic_regression_model.pkl       # Logistic regression model
│   ├── random_forest_model.pkl              # Random forest model
│   └── xgboost_model.pkl                    # XGBoost model
│
├── tests/                                    # Unit and integration tests
│   ├── __init__.py
│   └── test_data_loader.py                   # Tests for data loading functionality
│
├── docs/                                     # Project documentation
│   ├── __init__.py
│   └── project_overview.md                   # Detailed project documentation
│
├── main.py                                   # Entry point for script execution
├── structure.py                              # Project structure generator
├── data.py                                   # Data pipeline script
├── .gitignore                                # Git ignore rules
├── README.md                                 # Project overview and setup guide
├── requirements.txt
├── .gitattributes
├── Presentation.pdf
├── Presentation.pptx
└── infographic.png

                       # Python dependencies
```

---

## 🎯 Objectives

1. **Data Understanding**: Perform exploratory data analysis (EDA) to identify key features, distributions, and relationships that influence credit solvency.
2. **Feature Engineering**: Develop and apply preprocessing techniques to enhance model performance, including handling missing values, scaling, and encoding categorical variables.
3. **Model Development**: Train and evaluate multiple machine learning models (e.g., Logistic Regression, Random Forest, XGBoost) to predict credit solvency.
4. **Performance Evaluation**: Assess model performance using metrics such as accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
5. **Risk Segmentation**: Implement unsupervised clustering to segment users into risk bands based on consumption patterns and behavioral data.
6. **Visualization**: Generate comprehensive visualizations to communicate insights and model performance effectively.

---

## 🔧 Setup and Installation

### Prerequisites
- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/credit-solvency-assessment.git
   cd credit-solvency-assessment
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### Running the Project
- Execute the main script to run the entire pipeline:
  ```bash
  python main.py
  ```

- For modular execution, run individual notebooks or scripts in the `notebooks/` or `src/` directories.

### Key Scripts
- **`main.py`**: Orchestrates the end-to-end workflow, from data loading to model evaluation.
- **`structure.py`**: Generates and validates the project directory structure.
- **`data.py`**: Manages data ingestion, preprocessing, and feature engineering.

---

## 📊 Data Description

The dataset (`credit_solvency_dataset.csv`) contains the following types of features:
- **Demographic Information**: Age, income, employment status, etc.
- **Credit History**: Past loan behavior, credit utilization, payment history, etc.
- **Behavioral Data**: Spending patterns, transaction frequency, etc.
- **Target Variable**: Binary or multi-class label indicating credit solvency status.

---

## 🧠 Methodology

### 1. Data Exploration
- **Univariate Analysis**: Distribution of individual features.
- **Bivariate Analysis**: Relationships between pairs of features.
- **Correlation Analysis**: Identification of highly correlated features.
- **Target Analysis**: Exploration of the target variable across different segments.

### 2. Data Preprocessing
- **Handling Missing Values**: Imputation or removal of missing data.
- **Outlier Detection**: Identification and treatment of outliers.
- **Feature Scaling**: Standardization or normalization of numerical features.
- **Encoding**: Conversion of categorical variables into numerical format.

### 3. Model Training
- **Logistic Regression**: Baseline model for binary classification.
- **Random Forest**: Ensemble method for handling non-linear relationships.
- **XGBoost**: Gradient boosting framework for high performance.

### 4. Model Evaluation
- **Confusion Matrix**: True positives, false positives, true negatives, false negatives.
- **ROC Curve**: Trade-off between true positive rate and false positive rate.
- **Precision-Recall Curve**: Balance between precision and recall for imbalanced datasets.
- **Threshold Optimization**: Tuning decision thresholds for business objectives.

### 5. Unsupervised Clustering
- **Segmentation**: Grouping users based on digital content consumption patterns.
- **Risk Bands**: Assigning risk levels to clusters for targeted interventions.

---

## 📈 Results

The project generates the following outputs:
- **Model Performance Metrics**: Comparison of accuracy, precision, recall, and F1-score across models.
- **Visualizations**: Plots and dashboards for EDA, model evaluation, and risk segmentation.
- **Serialized Models**: Saved models for deployment or further analysis.

---

## 🛠️ Tools and Technologies

- **Programming Language**: Python
- **Libraries**:
  - Pandas, NumPy (Data Manipulation)
  - Scikit-learn (Machine Learning)
  - Matplotlib, Seaborn, Plotly (Visualization)
  - XGBoost (Gradient Boosting)
  - Jupyter (Interactive Notebooks)
- **Version Control**: Git

---

## 📂 Key Directories and Files

| Directory/File          | Purpose                                                                                     |
|-------------------------|---------------------------------------------------------------------------------------------|
| `data/`                 | Contains the raw and processed datasets.                                                    |
| `notebooks/`            | Jupyter notebooks for exploratory analysis and model development.                          |
| `src/`                  | Reusable Python modules for data loading, preprocessing, modeling, and visualization.     |
| `outputs/figures/`      | Generated visualizations and plots.                                                        |
| `models/`               | Serialized trained models for deployment.                                                  |
| `tests/`                | Unit and integration tests to ensure code reliability.                                     |
| `docs/`                 | Project documentation and detailed explanations.                                          |
| `main.py`               | Entry point for running the entire project pipeline.                                       |
| `requirements.txt`      | List of Python dependencies for the project.                                               |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

---

## 📜 License

This project is intended for portfolio and educational purposes.

## Author
MICHELL MATIAS TELLO
