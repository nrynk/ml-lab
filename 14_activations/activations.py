import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Load Dataset (MNIST digits)
print("Loading Digits dataset...")
digits = load_digits()
X = digits.data
y = digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 2. Compare Activation Functions
activation_functions = [
    ('Sigmoid (logistic)', 'logistic'),
    ('ReLU', 'relu'),
    ('Tanh', 'tanh')
]

results = []
print("\nTraining Neural Networks with Different Activations:")
for display_name, act in activation_functions:
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation=act,
        solver='adam',
        max_iter=300,
        random_state=42
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    results.append([display_name, round(acc, 4), round(training_time, 2)])
    print(f"Activation: {display_name:20s} | Accuracy: {acc*100:.2f}% | Training Time: {training_time:.2f}s")

# 3. Summary
res_df = pd.DataFrame(results, columns=['Activation Function', 'Test Accuracy', 'Training Time (s)'])
print("\n--- Activation Function Comparison Summary ---")
print(res_df.to_string(index=False))
