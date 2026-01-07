import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_size_vs_price(df: pd.DataFrame, feature: str, target: str = "SalePrice") -> None:
    """
    Plot regression plot showing relationship between a size feature and price.
    
    Args:
        df (pd.DataFrame): Housing data
        feature (str): Feature column name (e.g., "GrLivArea")
        target (str): Target column name (default: "SalePrice")
        
    Raises:
        ValueError: If required columns don't exist or aren't numeric
        Exception: If plotting fails
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("Cannot create plot from an empty DataFrame")
    
    if feature not in df.columns:
        raise ValueError(f"Feature column '{feature}' not found. Available columns: {list(df.columns)}")
    
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available columns: {list(df.columns)}")
    
    if not pd.api.types.is_numeric_dtype(df[feature]):
        raise ValueError(f"Feature column '{feature}' must be numeric")
    
    if not pd.api.types.is_numeric_dtype(df[target]):
        raise ValueError(f"Target column '{target}' must be numeric")
    
    try:
        # Remove NaN values for plotting
        plot_df = df[[feature, target]].dropna()
        
        if plot_df.empty:
            raise ValueError(f"No valid data points to plot after removing NaN values")
        
        plt.figure(figsize=(10, 6))
        sns.regplot(x=plot_df[feature], y=plot_df[target], scatter_kws={"alpha":0.4})
        plt.title(f"{feature} vs {target}")
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.tight_layout()
        plt.show()
        print(f"Regression plot '{feature} vs {target}' created successfully")
    
    except Exception as e:
        raise Exception(f"Error plotting {feature} vs {target}: {str(e)}")
