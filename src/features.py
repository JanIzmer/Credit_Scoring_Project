import numpy as np
import pandas as pd

def add_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered credit-related features to the dataset.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing credit data with columns like
                           'MonthlyIncome', 'DebtRatio', 'NumberOfOpenCreditLinesAndLoans',
                           'NumberOfDependents', 'age', and late payment counts.

    Returns:
        pd.DataFrame: A new DataFrame with additional features including:
                      - IncomePerCreditLine
                      - DebtPerDependent
                      - UtilizationPerCreditLine
                      - TotalLatePayments
                      - LatePaymentRatio
                      - SevereLateFlag
                      - CreditPerAge
                      - DebtRatioLog
                      - IncomeToDebtRatio
                      - DependentsFlag
                      - DependentsCategory
    """
    df = df.copy()
    
    # --- Income and credit lines ---
    df['IncomePerCreditLine'] = df.apply(
        lambda row: row['MonthlyIncome'] / row['NumberOfOpenCreditLinesAndLoans'] 
        if row['NumberOfOpenCreditLinesAndLoans'] > 0 else 0, axis=1)
    
    df['DebtPerDependent'] = df.apply(
        lambda row: row['DebtRatio'] / (row['NumberOfDependents'] + 1), axis=1)
    
    df['UtilizationPerCreditLine'] = df.apply(
        lambda row: row['RevolvingUtilizationOfUnsecuredLines'] / row['NumberOfOpenCreditLinesAndLoans']
        if row['NumberOfOpenCreditLinesAndLoans'] > 0 else 0, axis=1)
    
    # --- Late payments ---
    df['TotalLatePayments'] = (
        df['NumberOfTime30-59DaysPastDueNotWorse'] +
        df['NumberOfTime60-89DaysPastDueNotWorse'] +
        df['NumberOfTimes90DaysLate']
    )
    
    df['LatePaymentRatio'] = df.apply(
        lambda row: row['TotalLatePayments'] / row['NumberOfOpenCreditLinesAndLoans']
        if row['NumberOfOpenCreditLinesAndLoans'] > 0 else 0, axis=1)
    
    df['SevereLateFlag'] = df['NumberOfTimes90DaysLate'].apply(lambda x: 1 if x > 0 else 0)
    
    # --- Age features ---
    df['CreditPerAge'] = df.apply(
        lambda row: row['NumberOfOpenCreditLinesAndLoans'] / row['age'] 
        if row['age'] > 0 else 0, axis=1)
    
    # --- Financial load ---
    df['DebtRatioLog'] = df['DebtRatio'].apply(lambda x: np.log1p(x) if x > 0 else 0)
    df['IncomeToDebtRatio'] = df.apply(
        lambda row: row['MonthlyIncome'] / row['DebtRatio'] 
        if row['DebtRatio'] > 0 else 0, axis=1)
    
    # --- Dependents ---
    df['DependentsFlag'] = df['NumberOfDependents'].apply(lambda x: 1 if x > 0 else 0)
    df['DependentsCategory'] = df['NumberOfDependents'].apply(
        lambda x: 0 if x == 0 else 1 if x <= 2 else 2)
    
    return df