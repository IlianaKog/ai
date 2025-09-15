# -*- coding: utf-8 -*-
"""
Rule-based Sentiment Analysis

@author: iliana
"""

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


print("------------------------TextBlob------------------------------")
sentence_1 = "yesterday i had a great time at the movie it was really funny"
sentence_2 = "yesterday i had a great time at the movie but the parking was terrible"
sentence_3 = "yesterday i had a great time at the movie but the parking wasn't great"
sentence_4 = "i went to see a movie tomorrow"

sentences = [sentence_1,sentence_2,sentence_3,sentence_4]

for i, sentence in enumerate(sentences):
    sentiment_score = TextBlob(sentence)
    print(f"S{i+1}: {sentence} \npolarity: {sentiment_score.sentiment.polarity}\n")
    
#vader
print("------------------------VADER------------------------------")
vader = SentimentIntensityAnalyzer()

for i, sentence in enumerate(sentences):
    sentiment_score = vader.polarity_scores(sentence)
    print(f"S{i+1}: {sentence} \npolarity score : {sentiment_score}\n")