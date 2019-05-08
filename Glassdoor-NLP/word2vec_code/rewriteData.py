import pandas as pd
clean_questions = pd.read_csv("tokenized text data2.csv")
print(clean_questions.shape)
clean_questions.head()

import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize 
from gensim.models import Word2Vec 
from sklearn.decomposition import PCA
from matplotlib import pyplot

def getSentences(nlCol):
    # Create CBOW model 
    sentences=[]
    for s in clean_questions[nlCol]:# namestokenized_pros,tokenized_summary,tokenized_cons,tokenized_advice
        s=s[1:(len(s)-1)]
        s=s.replace("'",'')
        s=s.replace(" ",'')
        s=s.split(",")
        #print(s)
        sentences.append(s)
    return(sentences)

sentences=getSentences('tokenized_summary')
sentences.extend(getSentences('tokenized_pros'))
sentences.extend(getSentences('tokenized_cons'))
sentences.extend(getSentences('tokenized_advice'))

#clean_questions['tokenized_pros']=newArr
model = Word2Vec(sentences, min_count=1, size = 100, window = 5) # sg = 1???
#make word vec size 67529, but what meaning does that hold?

words = list(model.wv.vocab)
vocab_len=len(words)

# get avg of all word embeddings in a sample
def getEmbeds(nlCol):
    print('getting embeddings for '+nlCol)
    newAvgFeatures=[] # can change numfeatures added by modifying size above
    newMaxFeatures=[] # can change numfeatures added by modifying size above
    for s in clean_questions[nlCol]:# namestokenized_pros,tokenized_summary,tokenized_cons,tokenized_advice
        s=s[1:(len(s)-1)]
        s=s.replace("'",'')
        s=s.replace(" ",'')
        wordsi=s.split(",")#words in this sample
        #get avg embedding now
        avgembed=model[wordsi[0]]
        maxembed=model[wordsi[0]]
        maxembed.setflags(write=1)
        for i in range(1,len(wordsi)):
            nextembed=model[wordsi[i]]
            avgembed=np.add(avgembed,nextembed)
            for j in range(0,len(nextembed)):
                if nextembed[j] > maxembed[j]:
                    maxembed[j]=nextembed[j]
        avgembed=avgembed/len(wordsi)
        newAvgFeatures.append(avgembed)#add to full list
        newMaxFeatures.append(maxembed)
    return newMaxFeatures



# get rating outcomes
list_labels = clean_questions["overall-ratings"].tolist()
for i in range(0,len(clean_questions["overall-ratings"].tolist())):
    if i%1000==0:
        print(i)
    if clean_questions["overall-ratings"].tolist()[i] >= 3:
        list_labels[i] = 1
    else:
        list_labels[i] = 0

clean_questions['overall-ratings_binary'] = list_labels

# new max features has n x len(wordembedding for sample)
def addCols(newMaxFeatures, correspondingEmbedding):
    print('adding embeddings for '+nlCol)
    vecs=[]
    print('feature len'+str(len(newMaxFeatures[0])))
    for i in range(0, len(newMaxFeatures[0])): #for each component of word embedding 
        veci=[] # the ith component of each word wmedding
        for j in range(0, len(newMaxFeatures)):
            veci.append(newMaxFeatures[j][i]) # get ith component of jth sample
        vecs.append(veci)
        colname='wordEmbedding'+correspondingEmbedding+str(i)
        clean_questions[colname]=veci

nlCol='tokenized_summary'
addCols(getEmbeds(nlCol),nlCol)
nlCol='tokenized_pros'
addCols(getEmbeds(nlCol),nlCol)
nlCol='tokenized_cons'
addCols(getEmbeds(nlCol),nlCol)
nlCol='tokenized_advice'
addCols(getEmbeds(nlCol),nlCol)

clean_questions.to_csv('data_with_embeddings_sg.csv')

print(clean_questions.shape)
