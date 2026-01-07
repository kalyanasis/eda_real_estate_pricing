import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def build_sold_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a SoldDate column from YrSold and MoSold columns.
    
    Args:
        df (pd.DataFrame): Housing data with YrSold and MoSold columns
        
    Returns:
        pd.DataFrame: Data with SoldDate column added
        
    Raises:
        ValueError: If required columns are missing
    """
    required_cols = ["YrSold", "MoSold"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    try:
        df = df.copy()
        
        # Handle MoSold - can be numeric (1-12) or text ("Jan", "January")
        if df["MoSold"].dtype == "object":
            # Try short month format first
            try:
                df["MoSold"] = pd.to_datetime(df["MoSold"], format="%b").dt.month
            except (ValueError, TypeError):
                # Try full month format
                try:
                    df["MoSold"] = pd.to_datetime(df["MoSold"], format="%B").dt.month
                except (ValueError, TypeError):
                    # If both fail, try converting to numeric
                    df["MoSold"] = pd.to_numeric(df["MoSold"], errors="coerce")
        
        # Ensure numeric types
        df["YrSold"] = df["YrSold"].astype(int)
        df["MoSold"] = df["MoSold"].astype(int)
        
        # Validate month values
        if not df["MoSold"].between(1, 12).all():
            raise ValueError("MoSold values must be between 1 and 12")
        
        # Create date column
        df["SoldDate"] = pd.to_datetime(
            {"year": df["YrSold"], "month": df["MoSold"], "day": 1}
        )
        
        return df
    
    except Exception as e:
        raise Exception(f"Error building sold date: {str(e)}")

def plot_trends(df: pd.DataFrame) -> None:
    """
    Plot sales price trends over time (monthly and yearly).
    
    Args:
        df (pd.DataFrame): Housing data with YrSold and MoSold columns
        
    Raises:
        ValueError: If required columns are missing or data is invalid
        Exception: If plotting fails
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("Cannot plot trends from an empty DataFrame")
    
    required_cols = ["YrSold", "MoSold", "SalePrice"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    try:
        df = build_sold_date(df)
        
        # Monthly trend
        ts = df.groupby("SoldDate")["SalePrice"].median()
        
        if ts.empty:
            raise ValueError("No data available for trend plotting")
        
        plt.figure(figsize=(12, 5))
        ts.plot(title="Median SalePrice Over Time (Monthly)")
        plt.xlabel("Date")
        plt.ylabel("Median Sale Price")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Yearly trend
        ts_year = df.groupby("YrSold")["SalePrice"].median()
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=ts_year.index, y=ts_year.values, marker='o')
        plt.title("Yearly Median SalePrice")
        plt.xlabel("Year Sold")
        plt.ylabel("Median Price")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print("Market trend plots created successfully")
    
    except Exception as e:
        raise Exception(f"Error plotting market trends: {str(e)}")
