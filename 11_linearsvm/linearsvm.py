import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1. Load Dataset (Binary classification: Classes 0 and 1)
iris = datasets.load_iris()
X = iris.data[iris.target != 2, :2]
y = iris.target[iris.target != 2]

# 2. Train Linear SVM
model = SVC(kernel='linear', C=1.0)
model.fit(X, y)
y_pred = model.predict(X)

print(f"Linear SVM Accuracy: {accuracy_score(y, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=['Setosa', 'Versicolor']))

# 3. Decision Boundary & Support Vectors
w = model.coef_[0]
b = model.intercept_[0]
margin = 2 / np.linalg.norm(w)
print(f"Weight Vector w: [{w[0]:.4f}, {w[1]:.4f}]")
print(f"Intercept b:     {b:.4f}")
print(f"Margin Width:    {margin:.4f}")
print(f"Number of Support Vectors: {len(model.support_vectors_)}")

# 4. Plot Decision Boundary
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=60, label='Data Points')

plt.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    s=150,
    facecolors='none',
    edgecolors='green',
    linewidths=2,
    label='Support Vectors'
)

x_points = np.linspace(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2, 200)
y_decision = -(w[0] * x_points + b) / w[1]
y_margin_up = -(w[0] * x_points + b + 1) / w[1]
y_margin_down = -(w[0] * x_points + b - 1) / w[1]

plt.plot(x_points, y_decision, 'k-', linewidth=2, label='Decision Boundary')
plt.plot(x_points, y_margin_up, 'k--', linewidth=1, label='Margin (+1 / -1)')
plt.plot(x_points, y_margin_down, 'k--', linewidth=1)

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Linear SVM on Iris Dataset (Setosa vs Versicolor)")
plt.legend()
plt.tight_layout()
plt.savefig("linearsvm_plot.png", dpi=150)
print("\nPlot saved successfully to 'linearsvm_plot.png'.")
