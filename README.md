# Credit Scoring — Loan Default Prediction

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-success)
![Optuna](https://img.shields.io/badge/Tuning-Optuna-8f44ad)
![Power BI](https://img.shields.io/badge/Dashboard-Power%20BI-f2c811?logo=powerbi&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An end-to-end machine learning project that predicts the probability of a borrower
defaulting on a loan within two years, based on the Kaggle
[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset
(150,000 borrowers). The pipeline covers exploratory data analysis, feature
engineering, gradient-boosting model training with hyperparameter optimization,
threshold selection, and an interactive Power BI dashboard for business users.

> ⚠️ Educational portfolio project — not intended for real lending decisions.

## Results

All metrics are computed on a **20% hold-out test set** that was never used for
preprocessing, hyperparameter tuning or threshold selection.

| Model | Test ROC-AUC | Gini |
|---|---|---|
| Logistic Regression (baseline) | 0.864 | 0.727 |
| **LightGBM** (Optuna, 3-fold CV) | **0.868** | **0.736** |

At the operating threshold (0.585, selected on out-of-fold train predictions by
maximizing the F2-score) the model catches **71% of future defaulters** at 26%
precision — a deliberately recall-oriented trade-off, since missing a defaulter
is assumed to cost the bank more than reviewing a good client. The default rate
in the portfolio is ~6.7%.

| ROC curve | Feature importance |
|---|---|
| ![ROC curve](reports/figures/roc_curve.png) | ![Feature importance](reports/figures/feature_importance.png) |

**Key drivers of default risk** (by total gain): revolving credit utilization,
history of 90+ days delinquency, and the engineered `credit_per_age` — fully
consistent with the mutual-information ranking from the EDA.

## Project Workflow

### 1. Exploratory Data Analysis — [`01_EDA.ipynb`](notebooks/01_EDA.ipynb)
- Assessed missing values (`MonthlyIncome`, `NumberOfDependents`) and justified an imputation strategy.
- Analyzed feature distributions, heavy right-skew, and extreme outliers (e.g. revolving utilization values up to 50,708).
- Compared defaulters vs. non-defaulters with log-transformed histograms and boxplots.
- Ranked features with a correlation heatmap and mutual information against the target.

### 2. Modeling — [`02_Modeling.ipynb`](notebooks/02_Modeling.ipynb)
- **Leakage-free preprocessing:** the train/test split happens *before* any preprocessing; outlier-clipping bounds (95th percentile) and the income median are fitted on the training set only.
- **Missing values:** `MonthlyIncome` → train median + a binary `IncomeMissing` flag (0 would conflate "not reported" with real zero incomes); `NumberOfDependents` → 0.
- **Feature engineering:** `credit_per_age`, `late_payment_ratio`, `debt_to_income`; dropped a near-zero-signal feature (`NumberRealEstateLoansOrLines`).
- **Model:** LightGBM classifier; 50-trial [Optuna](https://optuna.org/) study maximizing mean ROC-AUC over a 3-fold stratified cross-validation on the training set.
- **Baseline:** logistic regression (industry-standard scorecard family) for an honest comparison.
- **Threshold selection:** on out-of-fold train predictions by maximizing F2 (recall-weighted).
- **Evaluation:** ROC-AUC / Gini, classification report and confusion matrix on the untouched hold-out set; feature importance by total gain.
- Generated predictions for the Kaggle test set.

### 3. Reporting — [`03_PowerBI_prepare.ipynb`](notebooks/03_PowerBI_prepare.ipynb) + [`Report.pbix`](reports/Report.pbix)
- Merged model predictions with client attributes into a BI-ready dataset.
- Built a Power BI dashboard prototype for exploring portfolio risk (open `reports/Report.pbix` in Power BI Desktop).

## Repository Structure

```
Credit_Scoring_Project/
├── data/
│   ├── raw/                    # Kaggle "Give Me Some Credit" dataset + data dictionary
│   └── processed/              # generated outputs (created by the notebooks, not tracked)
├── models/
│   ├── lgbm_model.pkl          # trained LightGBM model
│   └── best_threshold.json     # optimized classification threshold
├── notebooks/
│   ├── 01_EDA.ipynb            # exploratory data analysis
│   ├── 02_Modeling.ipynb       # feature engineering, training, evaluation
│   └── 03_PowerBI_prepare.ipynb# dataset preparation for the dashboard
├── reports/
│   ├── figures/                # exported result plots
│   └── Report.pbix             # Power BI dashboard
├── requirements.txt
└── README.md
```

## Getting Started

```bash
git clone https://github.com/JanIzmer/Credit_Scoring_Project.git
cd Credit_Scoring_Project

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook
```

Run the notebooks in order (`01` → `02` → `03`). The raw dataset is included in
`data/raw/`; all processed outputs are regenerated by the notebooks.
On Windows, `run_jupyter.bat` automates the environment setup.

## Dataset

150,000 borrowers, 10 features, binary target `SeriousDlqin2yrs`
(1 = experienced 90+ days delinquency within two years).

| Feature | Description |
|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | Balance on credit cards and personal lines of credit divided by total credit limits |
| `age` | Age of the borrower (years) |
| `NumberOfTime30-59DaysPastDueNotWorse` | Times 30–59 days past due in the last 2 years |
| `DebtRatio` | Monthly debt payments divided by monthly gross income |
| `MonthlyIncome` | Monthly income |
| `NumberOfOpenCreditLinesAndLoans` | Number of open loans and credit lines |
| `NumberOfTimes90DaysLate` | Times 90+ days past due |
| `NumberRealEstateLoansOrLines` | Number of mortgage and real estate loans |
| `NumberOfTime60-89DaysPastDueNotWorse` | Times 60–89 days past due in the last 2 years |
| `NumberOfDependents` | Number of dependents in the family |

## Tech Stack

**Python** (pandas, NumPy, scikit-learn, LightGBM, Optuna, Matplotlib, Seaborn) ·
**Jupyter Notebook** · **Power BI**

## Author

**Jan Izmer** — [github.com/JanIzmer](https://github.com/JanIzmer)

Built as a portfolio project to demonstrate data analysis and machine learning
skills in the credit-risk domain.
