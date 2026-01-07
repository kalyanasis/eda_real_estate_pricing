import pandas as pd
from real_estate_eda import data_loading, data_cleaning, feature_engineering
from real_estate_eda import univariate_analysis, multivariate_analysis
from real_estate_eda import size_impact, market_trends, clustering, baseline_model

df = data_loading.load_data("housing_data.csv")
df = data_cleaning.clean_data(df)
df = feature_engineering.engineer_features(df)   # <-- ensures BathsTotal exists

univariate_analysis.plot_distribution(df, "SalePrice")
multivariate_analysis.correlation_heatmap(df)
size_impact.plot_size_vs_price(df, "GrLivArea")
market_trends.plot_trends(df)

df = clustering.cluster_homes(df, ["GrLivArea","BathsTotal","GarageCars","TotalBsmtSF"], target="SalePrice")
results = baseline_model.train_baseline(df, ["GrLivArea","BathsTotal","GarageCars"], target="SalePrice")
print(results)
