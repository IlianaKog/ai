# -*- coding: utf-8 -*-
"""
Latent Semantic Analysis

@author: iliana
"""

import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from gensim import corpora
from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt


data = pd.read_csv("news_articles.csv")

# clean data
# content of the article --> lowercase and remove punctuation
articles = data['content'].str.lower().apply(lambda x: re.sub(r"([^\w\s])", "", x))
# stop word removal
en_stopwords = stopwords.words('english')
articles = articles.apply(lambda x: ' '.join([word for word in x.split() if word not in (en_stopwords)]))
# tokenize
articles = articles.apply(lambda x: word_tokenize(x))
# stemming (done for speed as we have a lot of text)
ps = PorterStemmer()
articles = articles.apply(lambda tokens: [ps.stem(token) for token in tokens])


# a dictionary of all words
dictionary = corpora.Dictionary(articles)
print(dictionary)
# vecotize - bag of words --> document term matrix
doc_term = [dictionary.doc2bow(text) for text in articles]

# -----------------------------LSA---------------------------

# specify num of topics
num_topics = 2
# LSA model
lsamodel = LsiModel(doc_term, num_topics=num_topics, id2word = dictionary) 
print(lsamodel.print_topics(num_topics=num_topics, num_words=5))


# find num of topics
# coherence scores
coherence_values = []
model_list = []

min_topics = 2
max_topics = 5

for num_topics_i in range(min_topics, max_topics+1):
    model = LsiModel(doc_term, num_topics=num_topics_i, id2word = dictionary, random_seed=0)
    model_list.append(model)
    coherence_model = CoherenceModel(model=model, texts=articles, dictionary=dictionary, coherence='c_v')
    coherence_values.append(coherence_model.get_coherence())
    print("next")
#plot
plt.plot(range(min_topics, max_topics+1), coherence_values)
plt.xlabel("num of topics")
plt.ylabel("coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

final_n_topics = 3
lsamodel_f = LsiModel(doc_term, num_topics=final_n_topics, id2word = dictionary) 
print(lsamodel_f.print_topics(num_topics=final_n_topics, num_words=5))

