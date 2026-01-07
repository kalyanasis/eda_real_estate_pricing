import pandas as pd
import os

def load_data(path: str) -> pd.DataFrame:
    """
    Load housing data from a CSV or Excel file.
    
    Args:
        path (str): Path to the data file (CSV or Excel format)
        
    Returns:
        pd.DataFrame: Loaded housing data
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file format is not supported
        Exception: If there's an error reading the file
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    
    try:
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        elif path.endswith(".xlsx"):
            df = pd.read_excel(path)
        else:
            raise ValueError("Unsupported file format. Please use CSV or XLSX files.")
        
        if df.empty:
            raise ValueError(f"The file {path} is empty.")
        
        print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns from {path}")
        return df
    
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file {path} is empty or corrupted.")
    except Exception as e:
        raise Exception(f"Error loading data from {path}: {str(e)}")
