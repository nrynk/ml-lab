import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Dataset
print("Loading Auto MPG dataset...")
try:
    from sklearn.datasets import fetch_openml
    auto = fetch_openml(name='autoMpg', version=1, as_frame=True)
    df = auto.frame
except Exception:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    cols = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
    df = pd.read_csv(url, delim_whitespace=True, names=cols, na_values='?')

df = df.dropna()
X = df[['displacement']].values.astype(float)
y = df['mpg'].values.astype(float)

# 2. Linear Regression Baseline
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)
mse_linear = mean_squared_error(y, y_pred_linear)
r2_linear = r2_score(y, y_pred_linear)

print("\n--- Linear Regression ---")
print(f"MSE:      {mse_linear:.4f}")
print(f"R2 Score: {r2_linear:.4f}")

# 3. Polynomial Regression (Degrees 2, 3, 4)
degrees = [2, 3, 4]
results = []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)
    y_pred_poly = poly_model.predict(X_poly)
    
    mse = mean_squared_error(y, y_pred_poly)
    r2 = r2_score(y, y_pred_poly)
    results.append([degree, mse, r2])
    print(f"\nPolynomial Degree = {degree}")
    print(f"MSE:      {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

# 4. Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.4, label='Actual Data')
X_range = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)

plt.plot(X_range, linear_model.predict(X_range), color='orange', linewidth=2, label='Linear Fit')

poly3 = PolynomialFeatures(degree=3)
poly3_model = LinearRegression()
poly3_model.fit(poly3.fit_transform(X), y)
plt.plot(X_range, poly3_model.predict(poly3.fit_transform(X_range)), color='red', linewidth=3, label='Polynomial Degree 3 Fit')

plt.xlabel("Engine Displacement")
plt.ylabel("Miles Per Gallon (MPG)")
plt.title("Polynomial Regression vs Linear Regression (Auto MPG)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("polynomial_plot.png", dpi=150)
print("\nPlot saved successfully to 'polynomial_plot.png'.")
