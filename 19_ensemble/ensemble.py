import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load Dataset
data_path = "Titanic.csv"
if not os.path.exists(data_path):
    data_path = os.path.join("..", "data", "Titanic.csv")

print(f"Loading Titanic dataset from {data_path}...")
data = pd.read_csv(data_path)

drop_cols = ['Cabin', 'Name', 'Ticket', 'PassengerId']
for c in drop_cols:
    if c in data.columns:
        data.drop(c, axis=1, inplace=True)

if 'Age' in data.columns:
    data['Age'] = data['Age'].fillna(data['Age'].median())
if 'Embarked' in data.columns:
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

data.dropna(inplace=True)

for col in ['Sex', 'Embarked']:
    if col in data.columns and data[col].dtype == 'object':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

target_col = 'Survived' if 'Survived' in data.columns else data.columns[0]
X = data.drop(target_col, axis=1)
y = data[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Bagging Ensemble
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    random_state=42
)
bagging.fit(X_train, y_train)
bag_pred = bagging.predict(X_test)

# 3. Boosting Ensemble (AdaBoost)
boosting = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    random_state=42
)
boosting.fit(X_train, y_train)
boost_pred = boosting.predict(X_test)

def get_metrics(y_true, y_pred):
    return [
        round(accuracy_score(y_true, y_pred), 4),
        round(precision_score(y_true, y_pred, zero_division=0), 4),
        round(recall_score(y_true, y_pred, zero_division=0), 4),
        round(f1_score(y_true, y_pred, zero_division=0), 4)
    ]

results = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Bagging (Random Forest-style)': get_metrics(y_test, bag_pred),
    'AdaBoost (Boosting)': get_metrics(y_test, boost_pred)
})

print("\n--- Ensemble Methods Comparison on Titanic Dataset ---")
print(results.to_string(index=False))
