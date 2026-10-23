import os
import time
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load Dataset
print("Loading Fashion MNIST dataset for SVM Kernel Comparison...")
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
X_train = (X_train / 255.0).reshape(len(X_train), -1)
X_test = (X_test / 255.0).reshape(len(X_test), -1)

train_size = 3000
test_size = 1000
X_tr, y_tr = X_train[:train_size], y_train[:train_size]
X_te, y_te = X_test[:test_size], y_test[:test_size]

# 2. Train & Evaluate Kernels
kernels = [
    ('Linear', SVC(kernel='linear', C=1.0, random_state=42)),
    ('Polynomial (deg=3)', SVC(kernel='poly', degree=3, C=1.0, random_state=42)),
    ('RBF', SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42))
]

results = []
for name, model in kernels:
    t0 = time.time()
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    dur = time.time() - t0
    
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_te, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_te, y_pred, average='weighted', zero_division=0)
    
    results.append([name, acc, prec, rec, f1, dur])
    print(f"[{name}] Accuracy: {acc*100:.2f}% | Time: {dur:.2f}s")

# 3. Comparison Summary
res_df = pd.DataFrame(
    results,
    columns=['Kernel Type', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Time (s)']
)
print("\n--- SVM Kernel Performance Comparison ---")
print(res_df.to_string(index=False))
