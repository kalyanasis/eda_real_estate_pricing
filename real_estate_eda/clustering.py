from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def cluster_homes(df: pd.DataFrame, features: list, target: str = "SalePrice", k: int = 4) -> pd.DataFrame:
    """
    Cluster homes using K-Means based on specified features.
    
    Args:
        df (pd.DataFrame): Housing data
        features (list): List of feature column names to use for clustering
        target (str): Target column for visualization (default: "SalePrice")
        k (int): Number of clusters (default: 4)
        
    Returns:
        pd.DataFrame: Data with 'Cluster' column added
        
    Raises:
        ValueError: If no valid features found or target missing
        Exception: If clustering fails
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("Cannot cluster an empty DataFrame")
    
    if not isinstance(features, list) or len(features) == 0:
        raise ValueError("Features must be a non-empty list")
    
    if k < 2:
        raise ValueError("Number of clusters (k) must be at least 2")
    
    if len(df) < k:
        raise ValueError(f"Cannot create {k} clusters with only {len(df)} data points")
    
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
        
        # Prepare data - handle missing values properly
        X = df[valid_features].copy()
        
        # Fill missing values with median for each column
        for col in valid_features:
            if X[col].isna().sum() > 0:
                median_val = X[col].median()
                if pd.isna(median_val):
                    # If median is NaN (all values are NaN), use 0
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(median_val)
        
        # Verify no NaN values remain
        if X.isna().any().any():
            raise ValueError("Unable to handle all missing values in features")
        
        # Perform clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(X)
        
        # Add cluster column to original dataframe
        df = df.copy()
        df["Cluster"] = cluster_labels
        
        # Visualization
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Cluster", y=target, data=df)
        plt.title(f"{target} Distribution by Cluster")
        plt.xlabel("Cluster")
        plt.ylabel(target)
        plt.tight_layout()
        plt.show()
        
        print(f"Clustering complete: Created {k} clusters using features {valid_features}")
        print(f"Cluster distribution: {df['Cluster'].value_counts().sort_index().to_dict()}")
        
        return df
    
    except Exception as e:
        raise Exception(f"Error during clustering: {str(e)}")
