import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.utils import resample

# 1. Load Dataset
iris = load_iris()
X = iris.data
y = iris.target
model = DecisionTreeClassifier(random_state=42)

# 2. Bootstrapping with OOB Evaluation
boot_accuracy = []
boot_f1 = []
n_iterations = 10

for i in range(n_iterations):
    indices = resample(range(len(X)), replace=True, random_state=i)
    X_boot, y_boot = X[indices], y[indices]
    
    oob = list(set(range(len(X))) - set(indices))
    if len(oob) == 0:
        continue
    X_test, y_test = X[oob], y[oob]
    
    model.fit(X_boot, y_boot)
    pred = model.predict(X_test)
    boot_accuracy.append(accuracy_score(y_test, pred))
    boot_f1.append(f1_score(y_test, pred, average='weighted'))

print("--- Bootstrapping Results (10 Iterations) ---")
print(f"Average Accuracy : {np.mean(boot_accuracy):.4f}")
print(f"Average F1 Score : {np.mean(boot_f1):.4f}")

# 3. 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracy = []
cv_f1 = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    cv_accuracy.append(accuracy_score(y_test, pred))
    cv_f1.append(f1_score(y_test, pred, average='weighted'))

print("\n--- 5-Fold Cross-Validation Results ---")
print(f"Average Accuracy : {np.mean(cv_accuracy):.4f}")
print(f"Average F1 Score : {np.mean(cv_f1):.4f}")

# 4. Resampling Comparison
res_df = pd.DataFrame({
    'Method': ['Bootstrapping (OOB)', '5-Fold Cross-Validation'],
    'Average Accuracy': [round(np.mean(boot_accuracy), 4), round(np.mean(cv_accuracy), 4)],
    'Average F1 Score': [round(np.mean(boot_f1), 4), round(np.mean(cv_f1), 4)]
})
print("\n--- Summary Comparison ---")
print(res_df.to_string(index=False))
