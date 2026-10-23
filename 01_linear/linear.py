import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Dataset
print("Loading California Housing dataset...")
housing = fetch_california_housing(as_frame=True)
X = housing.data[['AveRooms']].values
y = housing.target.values

# 2. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Linear Regression using Gradient Descent
m = len(y)
theta0 = 0.0
theta1 = 0.0
learning_rate = 0.01
iterations = 1000

for i in range(iterations):
    y_pred = theta0 + theta1 * X_scaled.flatten()
    d_theta0 = (1/m) * np.sum(y_pred - y)
    d_theta1 = (1/m) * np.sum((y_pred - y) * X_scaled.flatten())
    theta0 -= learning_rate * d_theta0
    theta1 -= learning_rate * d_theta1

y_pred_gd = theta0 + theta1 * X_scaled.flatten()
mse_gd = mean_squared_error(y, y_pred_gd)
r2_gd = r2_score(y, y_pred_gd)

print("\n--- Gradient Descent Results ---")
print(f"Intercept (theta0): {theta0:.4f}")
print(f"Slope (theta1):     {theta1:.4f}")
print(f"MSE:                {mse_gd:.4f}")
print(f"R2 Score:           {r2_gd:.4f}")

# 4. Linear Regression using Normal Equation
X_normal = np.c_[np.ones((m, 1)), X]
theta = np.linalg.inv(X_normal.T.dot(X_normal)).dot(X_normal.T).dot(y)
y_pred_ne = X_normal.dot(theta)
mse_ne = mean_squared_error(y, y_pred_ne)
r2_ne = r2_score(y, y_pred_ne)

print("\n--- Normal Equation Results ---")
print(f"Theta Values (Intercept, Slope): [{theta[0]:.4f}, {theta[1]:.4f}]")
print(f"MSE:                             {mse_ne:.4f}")
print(f"R2 Score:                        {r2_ne:.4f}")

# 5. Visualization
plt.figure(figsize=(8, 6))
plt.scatter(X, y, alpha=0.3, label='Actual Data', color='royalblue')
sorted_index = np.argsort(X.flatten())
plt.plot(
    X[sorted_index],
    y_pred_ne[sorted_index],
    color='red',
    linewidth=2,
    label='Regression Line (Normal Eq)'
)
plt.xlabel("Average Rooms (AveRooms)")
plt.ylabel("Median House Value ($100,000s)")
plt.title("Linear Regression on California Housing Dataset")
plt.legend()
plt.tight_layout()
plt.savefig("linear_plot.png", dpi=150)
print("\nPlot saved successfully to 'linear_plot.png'.")
