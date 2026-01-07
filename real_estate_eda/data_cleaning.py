import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean housing data by handling duplicates and missing values.
    
    Args:
        df (pd.DataFrame): Raw housing data
        
    Returns:
        pd.DataFrame: Cleaned housing data
        
    Raises:
        ValueError: If input is not a DataFrame or is empty
        Exception: If cleaning process fails
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("Cannot clean an empty DataFrame")
    
    try:
        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate rows")

        # Fill NA categories with 'NA'
        na_like_cats = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1",
                        "BsmtFinType2","FireplaceQu","GarageType","GarageFinish",
                        "GarageQual","GarageCond","PoolQC","Fence","MiscFeature"]
        
        filled_cols = 0
        for c in na_like_cats:
            if c in df.columns and df[c].isna().sum() > 0:
                df[c] = df[c].fillna("NA")
                filled_cols += 1

        # Numeric imputation
        numeric_cols_filled = 0
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
                numeric_cols_filled += 1
        
        print(f"Data cleaning complete: filled {filled_cols} categorical and {numeric_cols_filled} numeric columns")
        return df
    
    except Exception as e:
        raise Exception(f"Error during data cleaning: {str(e)}")
