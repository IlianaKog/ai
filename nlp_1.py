# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 16:52:52 2025

@author: User
"""

import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re
import pandas as pd

data = pd.read_csv("tripadvisor_hotel_reviews.csv")

data.info()
print(data.head())

print(data['Review'][0])

#new column 
data['review_lower_case'] = data['Review'].str.lower()
print(data.head())

#remove stop words
en_stopwords = stopwords.words('english')
en_stopwords.remove("not")
data['review_no_stopwords'] = data['review_lower_case'] .apply(lambda x: ' ' .join([word for word in x.split() if word not in (en_stopwords)]))

print(data['review_no_stopwords'][0])

#change star
data['review_no_stopwords_no_punct'] = data.apply(lambda x: re.sub(r"[*]","star", x['review_no_stopwords'] ), axis=1)

# remove punct
data['review_no_stopwords_no_punct'] = data.apply(lambda x: re.sub(r"[^\w\s]","", x['review_no_stopwords_no_punct']), axis = 1)

#Tokenizing
data['tokenized'] = data.apply(lambda x:
                               word_tokenize(x['review_no_stopwords_no_punct']), axis=1)

# Stemming
ps = PorterStemmer()
data['stemmed'] = data['tokenized'].apply(lambda tokens:
                               [ps.stem(token) for token in tokens])
    
# Lemma
lemmatizer = WordNetLemmatizer()
data['lemmetized'] = data["tokenized"].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])

#N grams
tokens_clean = sum(data['lemmetized'], [])
# n=1
unigrams = (pd.Series(nltk.ngrams(tokens_clean, 1)).value_counts()) 
print(unigrams)

# n=2
bigrams = (pd.Series(nltk.ngrams(tokens_clean, 2)).value_counts()) 
print(bigrams)

# n=4
ngrams = (pd.Series(nltk.ngrams(tokens_clean, 4)).value_counts()) 
print(ngrams)

