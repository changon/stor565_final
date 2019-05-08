import math
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

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
data= pd.read_csv("data_with_embeddings_sg.csv")

# take out na values
cols = ['overall-ratings', 'work-balance-stars' , 'culture-values-stars', 'carrer-opportunities-stars', 'comp-benefit-stars', 'senior-mangemnet-stars']
def check(input):
    return input.replace('.','',1).isdigit()

ratings = data[cols].astype(str)
ratingsIndex = ratings[ratings.applymap(check)].dropna(axis = 0).index
data=data.loc[ratingsIndex, :] #we could drop the rows where a rating is missing

print('data preprocessing')

# if want to include other covs. also don't need to remove none columns if not using other covs
toDelete=list(data.columns.values)[15:21]
toDelete.extend(list(data.columns.values)[0:8])
for colname in toDelete:
    del data[colname]

outcome = 'formerOrCurrent'
outcome_data=data[outcome]
del data[outcome]
#list(data.columns.values)

# partition data
print('data partitioning')
X_train, X_test, y_train, y_test = train_test_split(data, outcome_data, test_size=0.2,random_state=40)

# fit log reg.
print('fitting log reg')
Cs=[0.001, 0.01, 0.1, 1, 10, 100, 1000]
#cvlog = LogisticRegressionCV(Cs=Cs,cv=5,random_state=0,max_iter=4000).fit(X_train,y_train)
#cvlog.fit(X_train,y_train)
#vals=cvlog.scores_['former']
#cvReslog=avgCVscores(vals,len(Cs))

#cvReslog=[0.6297502925967751, 0.6297502874573433, 0.6294911327513637, 0.6293245277902383, 0.6293245277902383, 0.6293245277902383, 0.6293245277902383]

cvReslog=[0.6403222561682793, 0.6397115746615745, 0.6394297384192292, 0.6395002023064871, 0.6394532263816485, 0.6395002023064871, 0.6394532263816485]

# plot cvReslog
#fig = plt.figure()
#plt.plot(cvReslog)
#fig.suptitle('5 fold CV result: Log Reg', fontsize=20)
#plt.xlabel('Hyperparameters', fontsize=18)
#plt.ylabel('Accuracy', fontsize=16)
#fig.savefig('cvLogReg_review.jpg')

#bestInd=cvReslog.index(max(cvReslog))
#bestC=Cs[bestInd]

bestC=.001 # from cv results on just embeds
clf_w2v = LogisticRegression(C=bestC, class_weight='balanced', solver='newton-cg',
                         multi_class='ovr', random_state=40)
clf_w2v.fit(X_train, y_train)
y_predicted_word2vec = clf_w2v.predict(X_test)
displayMetrics(y_test,y_predicted_word2vec)

# fit rf
print('fitting rf')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

#rf = RandomForestClassifier(n_estimators=100)
#param_grid={'max_features': [4,10,math.sqrt(data.shape[1]),data.shape[1]/2]}#'mtry' in rf
#param_grid={'max_depth': [math.sqrt(data.shape[1]),data.shape[1]/2]}#'mtry' in rf
#cvResrf=GridSearchCV(rf, param_grid,cv=5)
#cvResrf.fit(X_train,y_train)
#cvrfscores=cvResrf.cv_results_['mean_test_score']

# want to plot cvrfscores
#cvrfscores=[0.86394684, 0.86385428]
# cvrfscores=[0.63961763, 0.61265472] #vals

#bestDepth=cvResrf.best_params_['max_depth']
bestDepth=20.0 # from cv results on just embeds
bestDepth=20.174241001832016

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
