# -*- coding: utf-8 -*-
"""
classifier - bag_of_words

@author: iliana
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer

dataset = load_dataset("Sp1786/multiclass-sentiment-analysis-dataset", split="train")

df = pd.DataFrame(dataset)
df["label"] = df["label"].map({0: "negative", 1: "neutral", 2:"positive"})

dfs = df.head(20)
dfs = dfs.sample(frac=1).reset_index(drop=True)

X = dfs['text']
y = dfs['sentiment']

#text vectorization
countvec = CountVectorizer()
countvec_fit = countvec.fit_transform(X)
bag_of_words = pd.DataFrame(countvec_fit.toarray(), columns = countvec.get_feature_names_out())


X_train, X_test, y_train, y_test = train_test_split(bag_of_words, y, test_size = 0.5, random_state = 7)


#Logistic Regression
lr = LogisticRegression(random_state=1).fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression:", accuracy_score(y_pred_lr, y_test))
#print(classification_report(y_test, y_pred_lr, zero_division=0))

#NaiveBayes
nb = MultinomialNB().fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print("NaiveBayes:",accuracy_score(y_pred_nb, y_test))

#Linear SVM
svm = SGDClassifier().fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("Linear SVM:", accuracy_score(y_pred_svm, y_test))

from sklearn.svm import SVC
gauss_svc = SVC(kernel='poly', C=100, gamma='scale', random_state=1)
gauss_svc.fit(X_train, y_train)
y_pred_gauss = gauss_svc.predict(X_test)

print("Polynomial SVM:", accuracy_score(y_test, y_pred_gauss))


