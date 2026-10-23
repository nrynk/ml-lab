import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# 1. Load or Generate Dataset
print("Loading / Synthesizing Customer Retail Data...")
np.random.seed(42)
n_customers = 500
customer_ids = np.arange(1001, 1001 + n_customers)
total_quantity = np.random.randint(1, 150, n_customers)
unit_price = np.random.uniform(5, 50, n_customers)
total_spending = total_quantity * unit_price
purchase_frequency = np.random.randint(1, 25, n_customers)

customer_data = pd.DataFrame({
    'CustomerID': customer_ids,
    'TotalQuantity': total_quantity,
    'TotalSpending': total_spending,
    'PurchaseFrequency': purchase_frequency
})

# 2. Customer Segmentation (Low, Medium, High)
customer_data['Segment'] = pd.qcut(
    customer_data['TotalSpending'],
    q=3,
    labels=['Low', 'Medium', 'High']
)

# 3. Features & Target
X = customer_data[['TotalQuantity', 'TotalSpending', 'PurchaseFrequency']]
y = customer_data['Segment']

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# 4. ID3 Decision Tree (Entropy Criterion)
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nID3 Decision Tree Accuracy: {accuracy * 100:.2f}%")

print("\n--- Feature Importances ---")
for feat, imp in zip(X.columns, model.feature_importances_):
    print(f"  {feat:20s}: {imp:.4f}")

# 5. Visualize Tree Structure
plt.figure(figsize=(12, 8))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=encoder.classes_,
    filled=True,
    rounded=True
)
plt.title("ID3 Decision Tree Structure for Customer Segmentation")
plt.tight_layout()
plt.savefig("decisiontree_plot.png", dpi=150)
print("\nTree plot saved successfully to 'decisiontree_plot.png'.")
