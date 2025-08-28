from  preprocessing import clean_credit_data, log_transform
from features import add_credit_features
import pandas as pd
import numpy as np

def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the dataset by cleaning, transforming, and adding features.

    Parameters:
        df (pd.DataFrame): Raw input DataFrame.

    Returns:
        pd.DataFrame: Processed DataFrame ready for modeling.
    """
    df = clean_credit_data(df)
    
    # Log transform skewed features
    skewed_cols = ['MonthlyIncome', 'DebtRatio', 'IncomePerCreditLine', 'DebtPerDependent',
    'UtilizationPerCreditLine', 'TotalLatePayments', 'CreditPerAge', 'IncomeToDebtRatio']
    df = log_transform(df, skewed_cols)
    
    df = add_credit_features(df)
    return df
# Not train decorator to exclude target variable
def not_train(func):
    def wrapper(df, *args, **kwargs):
        if 'SeriousDlqin2yrs' in df.columns:
            df = df.drop(columns=['SeriousDlqin2yrs'])
        return func(df, *args, **kwargs)
    return wrapper