# -*- coding: utf-8 -*-
"""
Named Entity Recognition

@author: iliana
"""

import spacy
from spacy import displacy
from spacy import tokenizer
import re
nlp = spacy.load('en_core_web_sm')
import wikipedia

wikipedia.set_lang('en')
title = "einstein"
find = wikipedia.search(title)
page = wikipedia.page(find[0])
title = page.title
wiki_text = wikipedia.summary(title)
print(wiki_text)

spacy_doc = nlp(wiki_text)

print("--------------------")
for word in spacy_doc.ents:
    print(word.text,word.label_)
    
#print("--------------------")
#displacy.render(spacy_doc,style="ent")

#clean and do the same
wiki_text_clean = re.sub(r'[^\w\s]', '', wiki_text).lower() # remove punctuation and lowercase
print(wiki_text_clean)
print("--------------------")
spacy_doc_clean = nlp(wiki_text_clean)
for word in spacy_doc_clean.ents:
    print(word.text,word.label_)