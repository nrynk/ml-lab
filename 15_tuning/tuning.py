import os
import time
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Load Dataset
print("Loading Fashion MNIST dataset for hyperparameter tuning...")
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

X_tr, y_tr = X_train[:3000], y_train[:3000]
X_te, y_te = X_test[:1000], y_test[:1000]

# 2. Hyperparameter Grid
learning_rates = [0.001, 0.01]
batch_sizes = [32, 64]
epochs_list = [10, 20]

results = []
print("\nRunning Hyperparameter Tuning Grid Search:")
for lr in learning_rates:
    for batch in batch_sizes:
        for epochs in epochs_list:
            t0 = time.time()
            model = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation='relu',
                learning_rate_init=lr,
                batch_size=batch,
                max_iter=epochs,
                solver='adam',
                random_state=42
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            acc = accuracy_score(y_te, y_pred)
            dur = time.time() - t0
            
            results.append([lr, batch, epochs, round(acc * 100, 2), round(dur, 2)])
            print(f"LR={lr:7.4f} | Batch={batch:2d} | Epochs={epochs:2d} -> Accuracy: {acc*100:.2f}% ({dur:.1f}s)")

# 3. Results Table
results_df = pd.DataFrame(
    results,
    columns=['Learning Rate', 'Batch Size', 'Epochs', 'Accuracy (%)', 'Time (s)']
)
results_df_sorted = results_df.sort_values(by='Accuracy (%)', ascending=False)

print("\n--- Hyperparameter Tuning Results (Ranked by Accuracy) ---")
print(results_df_sorted.to_string(index=False))
