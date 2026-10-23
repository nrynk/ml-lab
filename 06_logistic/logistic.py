import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load Dataset
column_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

data_path = "pima-indians-diabetes.csv"
if not os.path.exists(data_path):
    data_path = os.path.join("..", "data", "pima-indians-diabetes.csv")
if not os.path.exists(data_path):
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    df = pd.read_csv(url, names=column_names)
else:
    df = pd.read_csv(data_path, names=column_names)

# 2. Data Preprocessing (Impute zeroes in biological measurements)
columns_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns_with_zero:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Logistic Regression without Scaling
model_no_scaling = LogisticRegression(max_iter=1000, random_state=42)
model_no_scaling.fit(X_train, y_train)
y_pred_no = model_no_scaling.predict(X_test)

acc_no = accuracy_score(y_test, y_pred_no)
prec_no = precision_score(y_test, y_pred_no)
rec_no = recall_score(y_test, y_pred_no)
f1_no = f1_score(y_test, y_pred_no)

# 5. Logistic Regression with Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaling = LogisticRegression(max_iter=1000, random_state=42)
model_scaling.fit(X_train_scaled, y_train)
y_pred_scale = model_scaling.predict(X_test_scaled)

acc_scale = accuracy_score(y_test, y_pred_scale)
prec_scale = precision_score(y_test, y_pred_scale)
rec_scale = recall_score(y_test, y_pred_scale)
f1_scale = f1_score(y_test, y_pred_scale)

# 6. Results Comparison
results_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Without Scaling': [round(acc_no, 4), round(prec_no, 4), round(rec_no, 4), round(f1_no, 4)],
    'With Scaling':    [round(acc_scale, 4), round(prec_scale, 4), round(rec_scale, 4), round(f1_scale, 4)]
})

print("\n--- Performance Comparison: Impact of Feature Scaling ---")
print(results_df.to_string(index=False))
