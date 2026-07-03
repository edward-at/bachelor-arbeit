import pandas as pd
print(pd.read_parquet('/Users/edward/Downloads/dataset/regime_data/cluster_profile.parquet').round(3))