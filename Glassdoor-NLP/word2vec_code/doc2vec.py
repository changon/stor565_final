import math
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

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

def displayMetrics(y_test,y_predicted):
    accuracy_word2vec, precision_word2vec, recall_word2vec, f1_word2vec = get_metrics(y_test, y_predicted)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_word2vec, precision_word2vec, recall_word2vec, f1_word2vec))

def avgCVscores(scores,numparams):
    val=[0]*numparams
    for foldScores in scores:
        for i in range(0,numparams):
            val[i]+=foldScores[i]
    val[:] = [x / len(scores) for x in val]
    return(val)

# read in data and inital preprocessing
print('read in data')
data= pd.read_csv('tokenized text data2.csv')

#print('data preprocessing')
#toDelete=list(data.columns.values)[0:21] # up to binary ratings
#for colname in toDelete:
#    del data[colname]

#want 0-100 or summary vars
#names=list(data.columns.values)[101:401]
#for colname in names:
#    del data[colname]

#del data['formerOrCurrent']

#outcome = 'overall-ratings_binary_split2'
#outcome_data=data[outcome]
#del data[outcome]
#list(data.columns.values)

nls=['tokenized_pros','tokenized_summary', 'tokenized_cons', 'tokenized_advice']

docs=[]
for i in range(0,int(data.shape[0])):# namestokenized_pros,tokenized_summary,tokenized_cons,tokenized_advice
    curWords=[]
    for nlCol in nls:
        s=data[nlCol][i]
        s=s[1:(len(s)-1)]
        s=s.replace("'",'')
        s=s.replace(" ",'')
        s=s.split(",")
        curWords.extend(s)
        #print(s)
    docs.append(curWords)

documents=[TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]
size=50
model = Doc2Vec(documents, vector_size=size, window=2, min_count=1, workers=4)

embeds=[]
colnames=[]
for i in range(0,size):
    colnames.append('docEmbed'+str(i))

for i in range(0, data.shape[0]):
    embeds.append(list(model.docvecs[i]))

#company= pd.read_csv('tokenized text data2.csv')
#embedsdf=pd.DataFrame(embeds,columns=colnames)

#curData=pd.read_csv('data_with_embeddings_sg2.csv')
#newData=pd.concat([curData.reset_index(drop=True), embedsdf], axis=1)

newD= pd.read_csv('data_with_embeddings_sg2.csv')
toRm=list(newD.columns.values)[0:421]
for dname in toRm:
    del newD[dname]

newembeds=pd.concat([newD.reset_index(drop=True), embedsdf], axis=1)

#outcome='formerOrCurrent'
outcome='overall-ratings_binary_split2'
outcome_data=newembeds[outcome]
del newembeds[outcome]
del newembeds['formerOrCurrent']

inputdata = newembeds

# partition data
print('data partitioning')
X_train, X_test, y_train, y_test = train_test_split(inputdata, outcome_data, test_size=0.2,random_state=40)

# fit log reg.
print('fitting log reg')
#Cs=[0.001, 0.01, 0.1, 1, 10, 100, 1000]
#cvlog = LogisticRegressionCV(Cs=Cs,cv=5,random_state=0,max_iter=4000).fit(X_train,y_train)
#cvlog.fit(X_train,y_train)
#vals=cvlog.scores_[1]
#cvReslog=avgCVscores(vals,len(Cs))

cvReslog=[0.863965350088067, 0.863965350088067, 0.8639838566103482, 0.8639283267630553, 0.863410031054239, 0.8632619548882744, 0.8632619548882744]

# plot cvReslog
#fig = plt.figure()
#plt.plot(cvReslog)
#fig.suptitle('5 fold CV result: Log Reg', fontsize=20)
#plt.xlabel('Hyperparameters', fontsize=18)
#plt.ylabel('Accuracy', fontsize=16)
#fig.savefig('cvLogReg_review.jpg')

#bestInd=cvReslog.index(max(cvReslog))
#bestC=Cs[bestInd]

bestC=1000 # from cv results
clf_w2v = LogisticRegression(C=bestC, class_weight='balanced', solver='newton-cg',
                         multi_class='ovr', random_state=40)
clf_w2v.fit(X_train, y_train)
y_predicted_word2vec = clf_w2v.predict(X_test)
displayMetrics(y_test,y_predicted_word2vec)

# fit rf
print('fitting rf')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(n_estimators=100)
param_grid={'max_features': [4,10,math.sqrt(data.shape[1]),data.shape[1]/2]}#'mtry' in rf
param_grid={'max_depth': [math.sqrt(data.shape[1]),data.shape[1]/2]}#'mtry' in rf
cvResrf=GridSearchCV(rf, param_grid,cv=5)
cvResrf.fit(X_train,y_train)
cvrfscores=cvResrf.cv_results_['mean_test_score']

# want to plot cvrfscores
cvrfscores=[0.86394684, 0.86385428]

#bestDepth=cvResrf.best_params_['max_depth']
#bestDepth=20.0 # from cv

rf = RandomForestClassifier(n_estimators=100,max_depth=bestDepth)
rf.fit(X_train,y_train)
y_pred_rf=rf.predict(X_test)
displayMetrics(y_test,y_pred_rf)
#print(rf.feature_importances_) #most likely corresponding to colname order from list(data.columns.values)

# fit glmnet lasso
#print('fitting lasso')
#cvlasso = LogisticRegressionCV(Cs=Cs,penalty='l1',solver='saga',cv=5,random_state=0,max_iter=4000).fit(X_train,y_train)
#cvlasso.fit(X_train,y_train)
#vals=cvlasso.scores_[1]
#cvReslasso=avgCVscores(vals,len(Cs))
#
#bestInd=cvReslasso.index(max(cvReslasso))
#bestClasso=Cs[bestInd]
#
#lasso = LogisticRegression(C=bestClasso, class_weight='balanced', solver='saga', #or liblinear for small datasets
#                         multi_class='ovr',penalty='l1', random_state=40)
#lasso.fit(X_train, y_train)
#y_predicted_lasso= lasso.predict(X_test)
#displayMetrics(y_test,y_predicted_lasso)

# fit svm. https://scikit-learn.org/stable/modules/svm.html
#print('fitting svm')
#from sklearn import svm
#Gammas = [1e-3, 1e-4]
#svmcv= GridSearchCV(svm,
#            dict(C=Cs,
#                 gamma=Gammas),
#                 cv=2,
#                 pre_dispatch='1*n_jobs',
#                 n_jobs=1)

#svmcv= GridSearchCV(svm.SVC(kernel='rbf'),dict(C=[.01,10],gamma=[.1]),cv=5)

#svmcv.fit(X_train,y_train)

#scores = [x[1] for x in svmcv.grid_scores_]
#scores = np.array(scores).reshape(len(Cs), len(Gammas))

#for ind, i in enumerate(Cs):
#    plt.plot(Gammas, scores[ind], label='C: ' + str(i))
#plt.legend()
#plt.xlabel('Gamma')
#plt.ylabel('Mean score')
#plt.show()

#bestCsvm=cvsvmcv.best_params_['C']
#bestGamma=cvsvmcv.best_params_['gamma']
#bestCsvm=.1
#bestGamma=.1

#svm = svm.SVC(kernel='rbf',C=bestCsvm,gamma=bestGamma)
#svm.fit(X_train, y_train)
#y_pred_svm=svm.predict(X_test)
#displayMetrics(y_test,y_pred_svm)
