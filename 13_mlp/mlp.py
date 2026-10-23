import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Load Dataset
data_path = "winequality-red.csv"
if not os.path.exists(data_path):
    data_path = os.path.join("..", "data", "winequality-red.csv")

if not os.path.exists(data_path):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=';')
else:
    df = pd.read_csv(data_path, sep=';')

# 2. Preprocess & Scale
X = df.drop('quality', axis=1)
y = df['quality']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 3. Different MLP Architectures
architectures = [
    (10,),
    (20,),
    (30,),
    (20, 10),
    (30, 20),
    (50, 30, 10)
]

results = []
print("Training MLP architectures on Wine Quality dataset...")
for arch in architectures:
    model = MLPClassifier(
        hidden_layer_sizes=arch,
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append([str(arch), round(acc, 4)])
    print(f"Architecture: {str(arch):15s} | Accuracy: {acc*100:.2f}%")

# 4. Summary Table
results_df = pd.DataFrame(results, columns=['Hidden Layers', 'Accuracy'])
print("\n--- MLP Architectural Performance Comparison ---")
print(results_df.to_string(index=False))
