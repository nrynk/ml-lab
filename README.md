# Machine Learning Lab (PCCSL508)

Comprehensive implementation of all 20 experiments from the **ML Exam Pack Manual**. Each experiment is organized in its own self-contained folder featuring:
1. `algorithm.txt` – Plain-text Aim, Theory, and numbered Algorithm steps (easy to view with `cat`, `nano`, or `vim`).
2. `<name>.py` – Standalone Python script executable directly in terminal (`python3 <name>.py`).
3. `<name>.ipynb` – Formatted Jupyter Notebook with markdown documentation and code cells.
4. `output.txt` – Pre-computed verified console output and evaluation metrics.
5. `*_plot.png` – High-resolution visualization plots.

---

## 🚀 Quick Setup on Linux

Run the following command in your Linux terminal to clone the repository:

```bash
git clone https://github.com/nrynk/ml-lab.git
cd ml-lab
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🎯 The 5 Core Focus Programs

Here is the quick-reference guide for the five prioritized exam programs:

| # | Topic | Folder | 1-Word Name | Script | Notebook | Key Methods / Tasks |
|---|---|---|---|---|---|---|
| **1** | California Housing Linear Regression | `01_linear/` | `linear` | `linear.py` | `linear.ipynb` | Gradient Descent & Normal Equation; MSE & $R^2$; regression line plot |
| **2** | Diabetes Ridge & Lasso Regularization | `03_regularization/` | `regularization` | `regularization.py` | `regularization.ipynb` | RidgeCV & LassoCV hyperparameter tuning; feature selection |
| **3** | Pima Diabetes Logistic Regression | `06_logistic/` | `logistic` | `logistic.py` | `logistic.ipynb` | Binary classification; with & without `StandardScaler` scaling; Accuracy, Precision, Recall, F1 |
| **4** | Fashion MNIST KNN Image Classifier | `08_knn/` | `knn` | `knn.py` | `knn.ipynb` | Multi-class image classification; tested with $K \in [1, 3, 5, 7, 9]$; accuracy vs K plot |
| **5** | Fashion MNIST Neural Network Tuning | `15_tuning/` | `tuning` | `tuning.py` | `tuning.ipynb` | Grid search across learning rates ($0.001, 0.01$), batch sizes ($32, 64$), and epochs ($10, 20$) |

### How to Run the 5 Core Programs in Linux Terminal

```bash
# 1. Linear Regression
cd 01_linear && python3 linear.py

# 2. Ridge & Lasso Regularization
cd ../03_regularization && python3 regularization.py

# 3. Logistic Regression with Feature Scaling
cd ../06_logistic && python3 logistic.py

# 4. K-Nearest Neighbors on Fashion MNIST
cd ../08_knn && python3 knn.py

# 5. Neural Network Hyperparameter Tuning
cd ../15_tuning && python3 tuning.py
```

---

## 📚 Complete Index of All 20 Experiments

All 20 experiments from the manual are fully implemented and ready:

| Exp | Folder Name | Program Name | Dataset | Topic / Description |
|:---:|---|:---:|---|---|
| **01** | `01_linear` | `linear` | California Housing | Linear regression using Gradient Descent and Normal Equation |
| **02** | `02_polynomial` | `polynomial` | Auto MPG | Polynomial regression across degrees 2, 3, 4 vs linear regression |
| **03** | `03_regularization` | `regularization` | Diabetes | Ridge & Lasso regression with cross-validation tuning |
| **04** | `04_mle` | `mle` | Breast Cancer | Logistic regression parameter estimation using MLE and MAP (L1/L2) |
| **05** | `05_multinomial` | `multinomial` | 20 Newsgroups | Multinomial distribution parameter estimation via MLE and Dirichlet MAP |
| **06** | `06_logistic` | `logistic` | Pima Indians Diabetes | Logistic regression disease prediction with and without feature scaling |
| **07** | `07_naivebayes` | `naivebayes` | 20 Newsgroups | Text categorization using Multinomial Naïve Bayes vs Bernoulli Naïve Bayes |
| **08** | `08_knn` | `knn` | Fashion MNIST | Multi-class KNN classification experimenting with $K=1, 3, 5, 7, 9$ |
| **09** | `09_decisiontree` | `decisiontree` | Online Retail | ID3 Decision Tree (entropy) for customer segmentation |
| **10** | `10_income` | `income` | Adult Income | Comparison of Logistic Regression vs Decision Trees on tabular income data |
| **11** | `11_linearsvm` | `linearsvm` | Iris | Linear SVM boundary, maximum margin width, and support vector visualization |
| **12** | `12_kernelsvm` | `kernelsvm` | Fashion MNIST | Non-linear classification comparing Linear, Polynomial, and RBF kernels |
| **13** | `13_mlp` | `mlp` | Wine Quality | Multilayer Feed-Forward Network testing 6 hidden layer architectures |
| **14** | `14_activations` | `activations` | MNIST Digits | Comparing Sigmoid, ReLU, and Tanh activation convergence and accuracy |
| **15** | `15_tuning` | `tuning` | Fashion MNIST | Grid search tuning over learning rates, batch sizes, and training epochs |
| **16** | `16_clustering` | `clustering` | Mall Customers | K-Means Partitional vs Agglomerative Hierarchical Clustering comparison |
| **17** | `17_kmeans` | `kmeans` | Digits | K-Means clustering with Elbow Method (Inertia) and Silhouette Score |
| **18** | `18_resampling` | `resampling` | Iris | Bootstrapping with Out-Of-Bag (OOB) testing vs 5-Fold Cross-Validation |
| **19** | `19_ensemble` | `ensemble` | Titanic | Bagging (bootstrap aggregation) vs AdaBoost boosting ensemble methods |
| **20** | `20_biasvariance` | `biasvariance` | Housing | Bias-Variance Tradeoff curve across polynomial degrees 1 to 5 |

---

## 💻 Opening Notebooks & Text Files

- **Text Editor / Terminal:**
  ```bash
  cat 01_linear/algorithm.txt
  cat 01_linear/output.txt
  nano 01_linear/linear.py
  ```
- **Jupyter Notebook:**
  ```bash
  jupyter notebook
  ```
