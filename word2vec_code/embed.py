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

nlCol='tokenized_summary'

# Create CBOW model 
sentences=[]
for s in clean_questions[nlCol]:# namestokenized_pros,tokenized_summary,tokenized_cons,tokenized_advice
    s=s[1:(len(s)-1)]
    s=s.replace("'",'')
    s=s.replace(" ",'')
    s=s.split(",")
    #print(s)
    sentences.append(s)

#clean_questions['tokenized_pros']=newArr
model = Word2Vec(sentences, min_count=1, size = 100, window = 5,sg=1) # sg = 1???
#make word vec size 67529, but what meaning does that hold?

words = list(model.wv.vocab)
vocab_len=len(words)
#print(words)

# Print results 
print("Cosine similarity between 'growth' "+"and 'work' - CBOW : ", model.similarity('cultur', 'work'))

# word vector representation of cultur
print(model.wv['cultur'])

# words most similar to cultur
print(model.wv.most_similar('cultur'))

# result of semantically reasonable word vecs for cultur-worklif
print(model.wv.most_similar_cosmul(positive=['cultur'],negative=['worklif']))

# get odd word out. https://towardsdatascience.com/machine-learning-word-embedding-sentiment-classification-using-keras-b83c28087456

# use model.train(sentences, total_examples,epochs)


#print('print words')
#print(words)

# get avg of all word embeddings in a sample
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

# get rating outcomes
list_labels = clean_questions["overall-ratings"].tolist()
for i in range(0,len(clean_questions["overall-ratings"].tolist())):
    if i%1000==0:
        print(i)
    if clean_questions["overall-ratings"].tolist()[i] > 3:
        list_labels[i] = 1
    else:
        list_labels[i] = 0

clean_questions['overall-ratings_binary'] = list_labels

list_corpus=newMaxFeatures
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2,random_state=40)

# test log reg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

clf_w2v = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', 
                         multi_class='multinomial', random_state=40)
clf_w2v.fit(X_train, y_train)
y_predicted_word2vec = clf_w2v.predict(X_test)

def get_metrics(y_test, y_predicted):  
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,average='weighted')             
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,average='weighted')
    
    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    
    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1

accuracy_word2vec, precision_word2vec, recall_word2vec, f1_word2vec = get_metrics(y_test, y_predicted_word2vec)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_word2vec, precision_word2vec, recall_word2vec, f1_word2vec))

#visualize word embeddings
# fit a 2d PCA model to the vectors
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
