import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

#https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
def get_multiMetrics(y_test,y_predicted): #multi class case
    cm=confusion_matrix(y_test, y_predicted)
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    recall = TP/(TP+FN)
    spec = TN/(TN+FP)
    precision=TP/(TP+FP)
    ACC = (TP+TN)/(TP+FP+FN+TN)
    return ACC,precision, recall

def displayMetrics(y_test,y_predicted):
    accuracy_word2vec, precision_word2vec, recall_word2vec= get_multiMetrics(y_test, y_predicted)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f" % (accuracy_word2vec, precision_word2vec, recall_word2vec))


# read in data and inital preprocessing
print('read in data')
data= pd.read_csv("data_with_embeddings_sg.csv")

print('data preprocessing')
toDelete = list(data.columns.values)[0:2] # row nums repeated
toDelete.extend(list(data.columns.values)[3:23]) # up to binary ratings
for colname in toDelete:
    del data[colname]

#outcome = 'overall-ratings_binary'
outcome='company'
outcome_data=data[outcome]
del data[outcome]
#list(data.columns.values)

# partition data
print('data partitioning')
X_train, X_test, y_train, y_test = train_test_split(data, outcome_data, test_size=0.2,random_state=40)

# fit log reg.
from sklearn.metrics import confusion_matrix
print('fitting log reg')
clf_w2v = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                         multi_class='multinomial', random_state=40)
clf_w2v.fit(X_train, y_train)
y_predicted_word2vec = clf_w2v.predict(X_test)
confusion_matrix=confusion_matrix(y_test, y_predicted_word2vec)
displayMetrics(y_test,y_predicted_word2vec)

# fit rf
print('fitting rf')
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train,y_train)
y_pred_rf=rf.predict(X_test)
displayMetrics(y_test,y_pred_rf)
#print(rf.feature_importances_) #most likely corresponding to colname order from list(data.columns.values)

# fit glmnet lasso
print('fitting lasso')
#from sklearn.linear_model import Lasso
#lasso = Lasso(alpha=.1)
#lasso.fit(X_train,y_train)
#y_pred_lasso=lasso.predict(X_test)
#displayMetrics(y_test,y_pred_lasso)
#l1 indicates lasso
lasso = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                         multi_class='multinomial', random_state=40)
lasso.fit(X_train, y_train)
y_predicted_lasso= lasso.predict(X_test)
displayMetrics(y_test,y_predicted_lasso)

# fit svm. https://scikit-learn.org/stable/modules/svm.html
print('fitting svm')
from sklearn import svm
svm = svm.SVC(gamma='scale')
svm.fit(X_train, y_train)
y_pred_svm=svm.predict(X_test)
displayMetrics(y_test,y_predicted_svm)
