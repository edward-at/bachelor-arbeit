import pandas as pd, numpy as np
from pathlib import Path
A = Path("/Users/edward/Downloads/dataset/regime_data")
# 1. PCA variances cited in §9.3 (59.8 / 18.9 / 78.7)
ca = pd.read_parquet(A/"cluster_assignments.parquet")
print("cluster_assignments columns:", ca.columns.tolist())   # confirm pc1,pc2,pc3 exist
# 2. confirm the saved profile filename matches §9.6 "cluster_profile"
print("files in regime_data:", sorted(p.name for p in A.glob("*.parquet")))
print("PCA explained variance %:", np.round(pca.explained_variance_ratio_ * 100, 1))
print("PCA cumulative %        :", np.round(np.cumsum(pca.explained_variance_ratio_) * 100, 1))