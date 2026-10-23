import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load Dataset
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Preprocessing & Split
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return [name, round(acc, 4), round(prec, 4), round(rec, 4), round(f1, 4)]

# 3. MLE Logistic Regression (No penalty)
mle_model = LogisticRegression(penalty=None, max_iter=2000, random_state=42)
mle_model.fit(X_train, y_train)
mle_pred = mle_model.predict(X_test)

# 4. MAP with Gaussian Prior (L2 Penalty)
map_l2_model = LogisticRegression(penalty='l2', C=1.0, max_iter=2000, random_state=42)
map_l2_model.fit(X_train, y_train)
map_l2_pred = map_l2_model.predict(X_test)

# 5. MAP with Laplace Prior (L1 Penalty)
map_l1_model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)
map_l1_model.fit(X_train, y_train)
map_l1_pred = map_l1_model.predict(X_test)

# 6. Results Comparison
results = [
    evaluate("MLE (Unregularized)", y_test, mle_pred),
    evaluate("MAP (L2 - Gaussian Prior)", y_test, map_l2_pred),
    evaluate("MAP (L1 - Laplace Prior)", y_test, map_l1_pred)
]

df_results = pd.DataFrame(results, columns=["Estimation Method", "Accuracy", "Precision", "Recall", "F1 Score"])
print("\n--- Parameter Estimation Performance Comparison ---")
print(df_results.to_string(index=False))
