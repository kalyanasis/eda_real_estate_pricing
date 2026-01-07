import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_distribution(df: pd.DataFrame, column: str) -> None:
    """
    Plot the distribution of a numeric column with histogram and KDE.
    
    Args:
        df (pd.DataFrame): Housing data
        column (str): Name of the column to plot
        
    Raises:
        ValueError: If column doesn't exist or is not numeric
        Exception: If plotting fails
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("Cannot plot distribution of an empty DataFrame")
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame. Available columns: {list(df.columns)}")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric for distribution plot")
    
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column].dropna(), kde=True)
        plt.title(f"{column} Distribution")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()
        print(f"Distribution plot for '{column}' created successfully")
    
    except Exception as e:
        raise Exception(f"Error plotting distribution for '{column}': {str(e)}")
