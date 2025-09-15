# -*- coding: utf-8 -*-
"""
sentiment

@author: iliana
"""

import pandas as pd
import numpy as np
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import transformers
from transformers import pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from itertools import chain
from nltk import NaiveBayesClassifier

data = pd.read_csv("book_reviews_sample.csv")

#clean_data

data['review_text_cleaned'] = data['reviewText'].str.lower() # lower
data['review_text_cleaned'] = data.apply(lambda x: re.sub(r"([^\w\s])", "", x['review_text_cleaned']), axis=1) # remove punct

#vader
vader = SentimentIntensityAnalyzer()
data['vader_sentiment_score'] = data['review_text_cleaned'].apply(lambda review: vader.polarity_scores(review)['compound'])

#labels
bins = [-1, -0.2, 0.2, 1]
names = ['negative', 'neutral', 'positive']
#plot  
data['vader_sentiment_label'] = pd.cut(data['vader_sentiment_score'], bins, labels=names)
data['vader_sentiment_label'].value_counts().plot.bar()


