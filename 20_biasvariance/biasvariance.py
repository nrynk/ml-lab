import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Load Dataset
print("Loading housing dataset for Bias-Variance Tradeoff analysis...")
try:
    from sklearn.datasets import fetch_openml
    boston = fetch_openml(name="boston", version=1, as_frame=True)
    X = boston.data[['LSTAT']].values.astype(float)
    y = boston.target.values.astype(float)
except Exception:
    from sklearn.datasets import fetch_california_housing
    cal = fetch_california_housing(as_frame=True)
    X = cal.data[['MedInc']].values.astype(float)[:2000]
    y = cal.target.values.astype(float)[:2000]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Polynomial Degrees Analysis
degrees = [1, 2, 3, 4, 5]
train_error = []
validation_error = []

print("\nDegree | Training MSE | Validation MSE")
print("-" * 38)
for degree in degrees:
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    mse_train = mean_squared_error(y_train, train_pred)
    mse_val = mean_squared_error(y_test, test_pred)
    
    train_error.append(mse_train)
    validation_error.append(mse_val)
    print(f"  {degree:2d}   |   {mse_train:10.4f} |   {mse_val:10.4f}")

# 3. Plot Bias-Variance Tradeoff Curve
plt.figure(figsize=(9, 6))
plt.plot(degrees, train_error, label="Training Error (Bias)", marker='o', linewidth=2, color='blue')
plt.plot(degrees, validation_error, label="Validation Error (Variance)", marker='s', linewidth=2, color='red')
plt.xlabel("Polynomial Degree (Model Complexity)")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Bias-Variance Tradeoff in Polynomial Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("biasvariance_plot.png", dpi=150)
print("\nPlot saved successfully to 'biasvariance_plot.png'.")
