import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def correlation_heatmap(df: pd.DataFrame, target: str = "SalePrice") -> None:
    """
    Display a heatmap of correlations with the target variable.
    
    Args:
        df (pd.DataFrame): Housing data
        target (str): Target column name (default: "SalePrice")
        
    Raises:
        ValueError: If target column doesn't exist or no numeric columns found
        Exception: If plotting fails
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("Cannot create heatmap of an empty DataFrame")
    
    try:
        num_df = df.select_dtypes(include="number")
        
        if num_df.empty:
            raise ValueError("No numeric columns found in DataFrame")
        
        if target not in num_df.columns:
            raise ValueError(f"Target column '{target}' not found or not numeric. Available numeric columns: {list(num_df.columns)}")
        
        corr = num_df.corr()
        
        plt.figure(figsize=(10, 12))
        sns.heatmap(corr[[target]].sort_values(by=target, ascending=False),
                    annot=True, cmap="viridis", fmt=".2f", linewidths=0.5)
        plt.title(f"Correlations with {target}")
        plt.tight_layout()
        plt.show()
        print(f"Correlation heatmap for '{target}' created successfully")
    
    except Exception as e:
        raise Exception(f"Error creating correlation heatmap: {str(e)}")
