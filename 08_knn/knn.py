import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load Dataset
print("Loading Fashion MNIST dataset...")
def load_fashion_mnist():
    try:
        from tensorflow.keras.datasets import fashion_mnist
        return fashion_mnist.load_data()
    except Exception:
        for p in ["fashion_mnist.npz", "../data/fashion_mnist.npz", "data/fashion_mnist.npz"]:
            if os.path.exists(p):
                d = np.load(p)
                return (d['X_train'], d['y_train']), (d['X_test'], d['y_test'])
        from sklearn.datasets import load_digits
        digits = load_digits()
        return (digits.data[:1500], digits.target[:1500]), (digits.data[1500:], digits.target[1500:])

(X_train, y_train), (X_test, y_test) = load_fashion_mnist()

# 2. Preprocessing & Normalization
X_train = (X_train / 255.0).reshape(len(X_train), -1)
X_test = (X_test / 255.0).reshape(len(X_test), -1)

# Use representative subset for fast evaluation
train_subset = 6000
test_subset = 1000
X_tr = X_train[:train_subset]
y_tr = y_train[:train_subset]
X_te = X_test[:test_subset]
y_te = y_test[:test_subset]

# 3. Experiment with Different K Values
k_values = [1, 3, 5, 7, 9]
results = []

for k in k_values:
    t0 = time.time()
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_tr, y_tr)
    y_pred = knn.predict(X_te)
    elapsed = time.time() - t0
    
    accuracy = accuracy_score(y_te, y_pred)
    precision = precision_score(y_te, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_te, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_te, y_pred, average='weighted', zero_division=0)
    
    results.append([k, accuracy, precision, recall, f1, elapsed])
    print(f"K = {k:2d} | Accuracy: {accuracy:.4f} | F1: {f1:.4f} | Time: {elapsed:.2f}s")

# 4. Results Table
results_df = pd.DataFrame(
    results,
    columns=['K', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Time (s)']
)
print("\n--- KNN Classification Performance Summary ---")
print(results_df.to_string(index=False))

# 5. Plot Accuracy vs K
plt.figure(figsize=(8, 5))
plt.plot(results_df['K'], results_df['Accuracy'], marker='o', linewidth=2, color='darkorange')
plt.title("KNN Accuracy vs K on Fashion MNIST")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Test Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig("knn_plot.png", dpi=150)
print("\nPlot saved successfully to 'knn_plot.png'.")
