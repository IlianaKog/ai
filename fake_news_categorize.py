# -*- coding: utf-8 -*-
"""
Categorizing Fake News

@author: iliana
"""

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import spacy
from spacy import displacy
from spacy import tokenizer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gensim
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import LsiModel, TfidfModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report


data = pd.read_csv("fake_news_data.csv")

# plot num of fake and factual articles
data['fake_or_factual'].value_counts().plot(kind='bar')
plt.title('Count of Article Classification')
plt.ylabel('# of Articles')
plt.xlabel('Classification')


# POS tagging

nlp = spacy.load('en_core_web_sm')

# split data --> fake | factual 
fake_news = data[data['fake_or_factual'] == "Fake News"]
fact_news = data[data['fake_or_factual'] == "Factual News"]


# create spacey documents - use pipe for dataframe
fake_spaceydocs = list(nlp.pipe(fake_news['text']))
fact_spaceydocs = list(nlp.pipe(fact_news['text'])) 


