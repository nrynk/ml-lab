import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score

# 1. Load Dataset
categories = [
    'talk.religion.misc',
    'soc.religion.christian',
    'sci.space',
    'comp.graphics'
]
print("Loading 20 Newsgroups dataset for 4 categories...")
train_data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
test_data = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))

X_train, y_train = train_data.data, train_data.target
X_test, y_test = test_data.data, test_data.target

# 2. Multinomial Naive Bayes (Word Frequencies)
count_vectorizer = CountVectorizer(stop_words='english')
X_train_counts = count_vectorizer.fit_transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)

mnb = MultinomialNB()
mnb.fit(X_train_counts, y_train)
y_pred_mnb = mnb.predict(X_test_counts)

acc_mnb = accuracy_score(y_test, y_pred_mnb)
f1_mnb = f1_score(y_test, y_pred_mnb, average='weighted')

# 3. Bernoulli Naive Bayes (Binary Word Presence)
binary_vectorizer = CountVectorizer(stop_words='english', binary=True)
X_train_binary = binary_vectorizer.fit_transform(X_train)
X_test_binary = binary_vectorizer.transform(X_test)

bnb = BernoulliNB()
bnb.fit(X_train_binary, y_train)
y_pred_bnb = bnb.predict(X_test_binary)

acc_bnb = accuracy_score(y_test, y_pred_bnb)
f1_bnb = f1_score(y_test, y_pred_bnb, average='weighted')

# 4. Results Comparison
comparison = pd.DataFrame({
    'Model': ['Multinomial Naive Bayes', 'Bernoulli Naive Bayes'],
    'Feature Representation': ['Word Frequencies (Counts)', 'Binary Word Presence (0/1)'],
    'Accuracy': [round(acc_mnb, 4), round(acc_bnb, 4)],
    'F1 Score (Weighted)': [round(f1_mnb, 4), round(f1_bnb, 4)]
})

print("\n--- Naive Bayes Classifier Comparison ---")
print(comparison.to_string(index=False))
