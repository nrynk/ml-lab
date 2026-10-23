import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

# 1. Load Dataset
categories = ['sci.space', 'comp.graphics', 'rec.sport.baseball']
print("Loading 20 Newsgroups dataset...")
newsgroups = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))

# 2. Text Vectorization
vectorizer = CountVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(newsgroups.data)
words = np.array(vectorizer.get_feature_names_out())

# 3. Word Frequencies
word_counts = np.asarray(X.sum(axis=0)).flatten()
N = np.sum(word_counts)
K = len(word_counts)

# 4. MLE Estimation
theta_mle = word_counts / N

# 5. MAP Estimation with Dirichlet Priors
alpha_1 = 1.0
theta_map_1 = (word_counts + alpha_1) / (N + alpha_1 * K)

alpha_10 = 10.0
theta_map_10 = (word_counts + alpha_10) / (N + alpha_10 * K)

# 6. Display Top Words
top_indices = np.argsort(word_counts)[::-1][:10]
comparison_df = pd.DataFrame({
    'Word': words[top_indices],
    'Count': word_counts[top_indices],
    'MLE (alpha=0)': np.round(theta_mle[top_indices], 5),
    'MAP (alpha=1)': np.round(theta_map_1[top_indices], 5),
    'MAP (alpha=10)': np.round(theta_map_10[top_indices], 5)
})

print("\n--- Parameter Estimation for Top 10 Words ---")
print(comparison_df.to_string(index=False))

# 7. Visualization
plt.figure(figsize=(10, 6))
x_idx = np.arange(len(top_indices))
plt.bar(x_idx - 0.25, theta_mle[top_indices], width=0.25, label='MLE (alpha=0)', color='steelblue')
plt.bar(x_idx, theta_map_1[top_indices], width=0.25, label='MAP (alpha=1)', color='seagreen')
plt.bar(x_idx + 0.25, theta_map_10[top_indices], width=0.25, label='MAP (alpha=10)', color='coral')

plt.xticks(x_idx, words[top_indices], rotation=45)
plt.xlabel("Words")
plt.ylabel("Estimated Probability")
plt.title("Parameter Estimation: MLE vs MAP under Dirichlet Priors")
plt.legend()
plt.tight_layout()
plt.savefig("multinomial_plot.png", dpi=150)
print("\nPlot saved successfully to 'multinomial_plot.png'.")
