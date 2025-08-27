import pandas as pd
import numpy as np

def clean_credit_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # --- Target variable ---
    if 'SeriousDlqin2yrs' in df.columns:
        df['SeriousDlqin2yrs'] = df['SeriousDlqin2yrs'].astype(int)
    
    df = df[df['age'] >= 18]
    
    # --- Missing values and anomalies in income ---
    if 'MonthlyIncome' in df.columns:
        df['MonthlyIncome'] = df['MonthlyIncome'].fillna(0)
        df['MonthlyIncome'] = df['MonthlyIncome'].clip(lower=0)
    
    # --- Number of dependents ---
    if 'NumberOfDependents' in df.columns:
        df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0).astype(int)
    
    # --- Debt-related features ---
    debt_features = [
        'RevolvingUtilizationOfUnsecuredLines',
        'DebtRatio',
        'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfTimes90DaysLate',
        'NumberOfTime60-89DaysPastDueNotWorse'
    ]
    
    for col in debt_features:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            df[col] = df[col].clip(lower=0)
    
    # --- Credit lines and loans ---
    loan_features = ['NumberOfOpenCreditLinesAndLoans', 'NumberRealEstateLoansOrLines']
    for col in loan_features:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    
    return df