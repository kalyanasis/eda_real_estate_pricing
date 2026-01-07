import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing housing data columns.
    
    Features created:
    - PricePerSF: Sale price per square foot of living area
    - AgeAtSale: Age of house when sold
    - RemodelAge: Years since last remodel when sold
    - BathsTotal: Total bathrooms (full + 0.5*half)
    
    Args:
        df (pd.DataFrame): Cleaned housing data
        
    Returns:
        pd.DataFrame: Data with engineered features added
        
    Raises:
        ValueError: If input is not a DataFrame or is empty
        Exception: If feature engineering fails
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("Cannot engineer features on an empty DataFrame")
    
    try:
        features_created = []
        
        # Price per square foot
        if {"SalePrice","GrLivArea"}.issubset(df.columns):
            # Avoid division by zero
            df["PricePerSF"] = np.where(df["GrLivArea"] > 0, 
                                         df["SalePrice"] / df["GrLivArea"], 
                                         0)
            features_created.append("PricePerSF")

        # Age at sale
        if {"YrSold","YearBuilt"}.issubset(df.columns):
            df["AgeAtSale"] = df["YrSold"] - df["YearBuilt"]
            features_created.append("AgeAtSale")

        # Remodel age
        if {"YrSold","YearRemodAdd"}.issubset(df.columns):
            df["RemodelAge"] = df["YrSold"] - df["YearRemodAdd"]
            features_created.append("RemodelAge")

        # BathsTotal
        if {"FullBath","HalfBath"}.issubset(df.columns):
            df["BathsTotal"] = df["FullBath"] + 0.5 * df["HalfBath"]
            features_created.append("BathsTotal")
        
        if features_created:
            print(f"Created {len(features_created)} new features: {', '.join(features_created)}")
        else:
            print("Warning: No features could be created. Check if required columns exist.")
        
        return df
    
    except Exception as e:
        raise Exception(f"Error during feature engineering: {str(e)}")
