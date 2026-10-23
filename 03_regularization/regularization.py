import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# 2. Train-Test Split & Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Standard Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# 4. Ridge Regression with Cross-Validation
alphas = np.logspace(-3, 3, 100)
ridge = RidgeCV(alphas=alphas, cv=5)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

# 5. Lasso Regression with Cross-Validation
lasso = LassoCV(alphas=alphas, cv=5, max_iter=10000, random_state=42)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

# 6. Results Comparison
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge Regression', 'Lasso Regression'],
    'Best Alpha': ['N/A', round(ridge.alpha_, 4), round(lasso.alpha_, 4)],
    'MSE': [round(mse_lr, 2), round(mse_ridge, 2), round(mse_lasso, 2)],
    'R2 Score': [round(r2_lr, 4), round(r2_ridge, 4), round(r2_lasso, 4)]
})

print("\n--- Regression Model Performance Comparison ---")
print(results.to_string(index=False))

print("\nLasso Zeroed Coefficients (Feature Selection):")
for feature, coef in zip(diabetes.feature_names, lasso.coef_):
    status = "Zeroed Out (Ignored)" if abs(coef) < 1e-4 else f"Kept ({coef:.2f})"
    print(f"  {feature:10s} : {status}")
