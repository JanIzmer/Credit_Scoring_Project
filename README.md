# Credit Scoring Project

## Project Overview
This project aims to build a predictive model to estimate the probability of a client defaulting on a loan.  
- **Goal:** The Goal is to create ML model which can predict the probabilyty of a client defaulting binary outcome (`0` — repaid, `1` — default).  
- **Business Value:** Helps the bank reduce credit risk and make informed lending decisions. *In this project only for educational purpose*.

## Dataset
The dataset was obtained from the open-source platform Kaggle. It contains information about clients and their loan histories.

**Features include:**
- `SeriousDlqin2yrs` — target variable (1 if defaulted within 2 years, 0 if repaid)
- `RevolvingUtilizationOfUnsecuredLines` — ratio of credit line utilization
- `age` — age of the client
- `NumberOfTime30-59DaysPastDueNotWorse` — times 30-59 days past due
- `DebtRatio` — monthly debt payments / monthly income
- `MonthlyIncome` — income of the client
- `NumberOfOpenCreditLinesAndLoans` — active credit lines
- `NumberOfTimes90DaysLate` — times 90+ days past due
- `NumberRealEstateLoansOrLines` — number of real estate loans
- `NumberOfTime60-89DaysPastDueNotWorse` — times 60-89 days past due
- `NumberOfDependents` — number of dependents

## Project Steps

### 1. Project Layout Creation
- Defined the project structure and workflow

### 2. Task Breakdown
- Split the project into sequential tasks from data preprocessing to model evaluation and reporting.

### 3. Dataset Acquisition and Preparation
- Acquired dataset with client demographic, financial, and credit history information.
- Checked dataset size.
- Handled missing values, outliers, and duplicates.
- Processed categorical features (e.g., profession, gender, education).

### 4. Exploratory Data Analysis (EDA)
- Analyzed feature distributions.
- Compared defaulters vs. non-defaulters.
- Investigated correlations and key factors affecting credit risk.

### 5. Feature Engineering and Data Preparation
- Created new features, such as debt-to-income ratio and age bins.
- Applied log-transformations to skewed features.

### 6. Model Training
- Trained a LightGBM model on engineered features.
- Handled class imbalance using SMOTE.
- Performed train-test split.
- Optimized hyperparameters using Optuna.
- Selected the best probability threshold for classification.

### 7. Model Testing and Evaluation
- Evaluated model performance using ROC-AUC, F1-score, Precision, and Recall.
- Analyzed feature importance and SHAP values for interpretability.

## Results
- Best model: `LightGBM` with ROC-AUC = 0.86,
- Key risk factors:
    - Late payment ratio
    - Credit per age
    - Age

## Demo
A prototype PowerBi dashboard reports/report.pbix

## Repository Structure
```
.
├── data/              # Datasets
├── notebooks/         # Jupyter Notebooks (EDA, models)
├── src/               # ML pipeline, functions
├── reports/           # Reports and presentations
└── README.md          # Project description
```

## Tech Stack
- Python (pandas, numpy, scikit-learn, lightgbm, smoteen, optuna),
- Jupyter Notebook,
- Matplotlib / Seaborn ,
- Power BI (for visualization).

## Author
This project was created for educational purposes to demonstrate Data Analytics / Data Science skills in the financial sector.
