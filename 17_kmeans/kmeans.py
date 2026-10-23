import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load Dataset
print("Loading Digits dataset for K-Means analysis...")
digits = load_digits()
X = digits.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Iterate K from 2 to 10
k_values = range(2, 11)
inertia = []
silhouette = []

print("\nK\tInertia\t\tSilhouette Score")
print("-" * 35)
for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    inertia.append(model.inertia_)
    score = silhouette_score(X_scaled, labels)
    silhouette.append(score)
    print(f"{k}\t{model.inertia_:.2f}\t\t{score:.4f}")

# 3. Visualization: Elbow & Silhouette Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(k_values, inertia, marker='o', linewidth=2, color='crimson')
ax1.set_title("Elbow Method for Optimal K (Digits)")
ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("Inertia (Sum of Squared Distances)")
ax1.grid(True)

ax2.plot(k_values, silhouette, marker='s', linewidth=2, color='navy')
ax2.set_title("Silhouette Score vs K (Digits)")
ax2.set_xlabel("Number of Clusters (K)")
ax2.set_ylabel("Silhouette Score")
ax2.grid(True)

plt.tight_layout()
plt.savefig("kmeans_plot.png", dpi=150)
print("\nPlot saved successfully to 'kmeans_plot.png'.")
