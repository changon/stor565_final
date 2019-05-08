import pandas as pd
clean_questions = pd.read_csv("tokenized text data2.csv")
clean_questions.head()

import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize 
from gensim.models import Word2Vec 
from sklearn.decomposition import PCA
from matplotlib import pyplot

#print(list(clean_questions.columns.values))

#print(clean_questions['tokenized_pros']) 

# Create CBOW model 
sentences=[]
for s in clean_questions['tokenized_pros']:# namestokenized_pros,tokenized_summary,tokenized_cons,tokenized_advice
    s=s[1:(len(s)-1)]
    s=s.replace("'",'')
    s=s.replace(" ",'')
    s=s.split(",")
    #print(s)
    sentences.append(s)

#clean_questions['tokenized_pros']=newArr
model = Word2Vec(sentences, min_count=1, size = 100, window = 5) # sg = 1???
#make word vec size 67529, but what meaning does that hold?

X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]), fontsize=5)
pyplot.show()
# rescale or only keep certain words??
