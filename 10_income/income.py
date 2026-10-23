import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 1. Load Dataset
data_path = "adult.csv"
if not os.path.exists(data_path):
    data_path = os.path.join("..", "data", "adult.csv")

print(f"Loading Adult Income dataset from {data_path}...")
df = pd.read_csv(data_path)

df.columns = [c.strip().lower() for c in df.columns]
target_col = 'income' if 'income' in df.columns else df.columns[-1]

df.replace(" ?", np.nan, inplace=True)
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str).str.strip())

X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

# 3. Decision Tree
dt_model = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

def get_metrics(y_true, y_pred):
    return [
        round(accuracy_score(y_true, y_pred), 4),
        round(precision_score(y_true, y_pred, zero_division=0), 4),
        round(recall_score(y_true, y_pred, zero_division=0), 4),
        round(f1_score(y_true, y_pred, zero_division=0), 4)
    ]

results = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Logistic Regression': get_metrics(y_test, lr_pred),
    'Decision Tree (depth=5)': get_metrics(y_test, dt_pred)
})

print("\n--- Performance Comparison: Logistic Regression vs Decision Tree ---")
print(results.to_string(index=False))
