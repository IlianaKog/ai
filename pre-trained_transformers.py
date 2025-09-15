# -*- coding: utf-8 -*-
"""
pre-trained transfromers

@author: iliana
"""

import transformers

from transformers import pipeline

sentiment_pipeline = pipeline('sentiment-analysis', model="finiteautomata/bertweet-base-sentiment-analysis")

sentence_1 = "yesterday i had a great time at the movie it was really funny"
sentence_2 = "yesterday i had a great time at the movie but the parking was terrible"
sentence_3 = "yesterday i had a great time at the movie but the parking wasn't great"
sentence_4 = "i went to see a movie tomorrow"

test1 = sentiment_pipeline(sentence_1)
print(test1)
print([sub['label'] for sub in test1])

test2 = sentiment_pipeline(sentence_2)
print(test2)
print([sub['label'] for sub in test2])

test3 = sentiment_pipeline(sentence_3)
print(test3)
print([sub['label'] for sub in test3])

test4 = sentiment_pipeline(sentence_4)
print(test4)
print([sub['label'] for sub in test4])