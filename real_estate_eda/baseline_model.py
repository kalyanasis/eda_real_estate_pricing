from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np

def train_baseline(df: pd.DataFrame, features: list, target: str = "SalePrice") -> dict:
    """
    Train a baseline Linear Regression model and return evaluation metrics.
    
    Args:
        df (pd.DataFrame): Housing data
        features (list): List of feature column names
        target (str): Target column name (default: "SalePrice")
        
    Returns:
        dict: Dictionary containing R2, MAE, RMSE, and feature coefficients
        
    Raises:
        ValueError: If no valid features found or data is invalid
        Exception: If training fails
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("Cannot train model on an empty DataFrame")
    
    if not isinstance(features, list) or len(features) == 0:
        raise ValueError("Features must be a non-empty list")
    
    try:
        # Validate features
        valid_features = [f for f in features if f in df.columns]
        invalid_features = [f for f in features if f not in df.columns]
        
        if invalid_features:
            print(f"Warning: Ignoring invalid features: {invalid_features}")
        
        if not valid_features:
            raise ValueError(f"No valid features found. Requested: {features}, Available: {list(df.columns)}")
        
        # Check target exists
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame")
        
        # Prepare data
        X = df[valid_features].copy()
        y = df[target].copy()
        
        # Remove rows with NaN in target
        valid_indices = ~y.isna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(y) == 0:
            raise ValueError("No valid target values found")
        
        # Handle missing values in features
        for col in valid_features:
            if X[col].isna().sum() > 0:
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val if not pd.isna(median_val) else 0)
        
        # Split data
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if len(Xtr) == 0 or len(Xte) == 0:
            raise ValueError("Insufficient data for train-test split")
        
        # Train model
        model = LinearRegression()
        model.fit(Xtr, ytr)
        
        # Make predictions
        pred_train = model.predict(Xtr)
        pred_test = model.predict(Xte)
        
        # Calculate metrics
        r2_train = r2_score(ytr, pred_train)
        r2_test = r2_score(yte, pred_test)
        mae = mean_absolute_error(yte, pred_test)
        rmse = np.sqrt(mean_squared_error(yte, pred_test))
        
        # Feature importance (coefficients)
        feature_importance = dict(zip(valid_features, model.coef_))
        
        results = {
            "R2_train": round(r2_train, 4),
            "R2_test": round(r2_test, 4),
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "Intercept": round(float(model.intercept_), 2),
            "Feature_Coefficients": {k: round(v, 2) for k, v in feature_importance.items()},
            "Features_Used": valid_features,
            "Training_Samples": len(Xtr),
            "Test_Samples": len(Xte)
        }
        
        print(f"\nBaseline Model Training Complete:")
        print(f"  Features used: {valid_features}")
        print(f"  R² (train): {results['R2_train']:.4f}")
        print(f"  R² (test): {results['R2_test']:.4f}")
        print(f"  MAE: ${results['MAE']:,.2f}")
        print(f"  RMSE: ${results['RMSE']:,.2f}")
        
        return results
    
    except Exception as e:
        raise Exception(f"Error during model training: {str(e)}")
