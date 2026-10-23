import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# 1. Load Dataset
data_path = "Mall_Customers.csv"
if not os.path.exists(data_path):
    data_path = os.path.join("..", "data", "Mall_Customers.csv")

print(f"Loading Mall Customers dataset from {data_path}...")
df = pd.read_csv(data_path)

X = df.iloc[:, [3, 4]].values

# 2. K-Means Clustering (K=5)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)
kmeans_sil = silhouette_score(X, kmeans_labels)

# 3. Agglomerative Hierarchical Clustering (K=5, linkage='ward')
agg = AgglomerativeClustering(n_clusters=5, linkage='ward')
agg_labels = agg.fit_predict(X)
agg_sil = silhouette_score(X, agg_labels)

print("\n--- Clustering Evaluation (Silhouette Score) ---")
print(f"K-Means Clustering:       {kmeans_sil:.4f}")
print(f"Agglomerative Clustering: {agg_sil:.4f}")

# 4. Visualization Side-by-Side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='tab10', s=50, alpha=0.8)
ax1.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200, c='black', marker='X', label='Centroids'
)
ax1.set_title(f"K-Means Clustering (Silhouette: {kmeans_sil:.3f})")
ax1.set_xlabel("Annual Income (k$)")
ax1.set_ylabel("Spending Score (1-100)")
ax1.legend()
ax1.grid(True)

ax2.scatter(X[:, 0], X[:, 1], c=agg_labels, cmap='tab10', s=50, alpha=0.8)
ax2.set_title(f"Agglomerative Clustering (Silhouette: {agg_sil:.3f})")
ax2.set_xlabel("Annual Income (k$)")
ax2.set_ylabel("Spending Score (1-100)")
ax2.grid(True)

plt.tight_layout()
plt.savefig("clustering_plot.png", dpi=150)
print("\nPlot saved successfully to 'clustering_plot.png'.")
