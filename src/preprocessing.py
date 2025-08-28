import pandas as pd
import numpy as np

def clean_credit_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and preprocesses a credit dataset for analysis or modeling.

    Parameters:
        df (pd.DataFrame): Input dataset containing credit-related features.
    
    Returns:
        pd.DataFrame: A cleaned copy of the input dataset.
    """

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

def log_transform(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Applies log transformation to specified numerical columns to reduce skewness.
    
    The transformation is performed as log(1 + x). Zero or negative values are kept as 0.
    
    Parameters:
        df (pd.DataFrame): Input dataset containing numerical features.
        columns (list): List of column names to apply the log transformation.
    
    Returns:
        pd.DataFrame: A copy of the dataset with specified columns log-transformed.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: np.log1p(x) if x > 0 else 0)
    return df
