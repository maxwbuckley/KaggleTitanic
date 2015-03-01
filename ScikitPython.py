##TODO Tidy this mess
import csv
import sklearn as sk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import cross_val_score, LeaveOneOut
from scipy.stats import sem
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn import linear_model
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.ensemble import AdaBoostRegressor

columnList = ['Sex', 'Pclass', 'Title', 'SibSp','Parch' ]

def prepDataSet(columnList):
  """ Returns our trainingset all vectorized and ready for use, our testingset
      and our trainingset labels """

  pdtrainframe = pd.io.parsers.read_csv('processedcsv/train.csv')
  pdtestframe = pd.io.parsers.read_csv('processedcsv/test.csv')
  titanic_y = pdtrainframe['Survived']
  pdtrainframe = pdtrainframe.drop('Survived',1)

  # This will make sure we have the same columns in both frames. I have already
  # removed NA's in the R script.
  assert(list(pdtrainframe.columns.values)==list(pdtestframe.columns.values))

  trainingFrame = pdtrainframe[columnList]
  testingFrame = pdtestframe[columnList]

  trainingFrame_dict = trainingFrame.T.to_dict().values()
  testingFrame_dict = testingFrame.T.to_dict().values()

  vectorizer = DV(sparse = False)
  trainingFrame = vectorizer.fit_transform(trainingFrame_dict)
  testingFrame = vectorizer.transform(testingFrame_dict)
  return(trainingFrame, testingFrame, titanic_y)




def loo_cv(X_train,y_train,clf):
  """ Leave one out cross validation """
  loo = LeaveOneOut(X_train[:].shape[0])
  scores=np.zeros(X_train[:].shape[0])
  for train_index,test_index in loo:
      X_train_cv, X_test_cv= X_train[train_index], X_train[test_index]
      y_train_cv, y_test_cv= y_train[train_index], y_train[test_index]
      clf = clf.fit(X_train_cv,y_train_cv)
      y_pred=clf.predict(X_test_cv)
      scores[test_index]=metrics.accuracy_score(y_test_cv.astype(int), y_pred.astype(int))
  print ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))


def writeOutput(predictionVector, name):
  pdtestframe = pd.io.parsers.read_csv('processedcsv/test.csv')
  output = pd.concat([pdtestframe['PassengerId'],pd.DataFrame({'Survived':predictionVector})], axis=1)
  output.to_csv("outputcsv/"+name+".csv",index=False)  
  return output

def trainMediumModels(dataset, datasettest, traininglabels):
  titanic_X = dataset
  titanic_X_test = datasettest
  # print titanic_X_test
  
  
  ginitree = tree.DecisionTreeClassifier(criterion='gini', max_depth=4,
                                    min_samples_leaf=5)
  ginitree = ginitree.fit(titanic_X, traininglabels)
  
  entropytree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4,
                                    min_samples_leaf=5)

  entropytree = entropytree.fit(titanic_X, traininglabels)


  #print "Trying Decision Tree:"
  #loo_cv(titanic_X, traininglabels,ginitree)
  #treepred = ginitree.predict(titanic_X_test)
  """"""
  adaginitree = AdaBoostRegressor(ginitree, n_estimators = 50, random_state = 33)  

  adaentropytree = AdaBoostRegressor(entropytree, n_estimators = 50, 
                                     random_state = 33)

  logreg = linear_model.LogisticRegression(C = 1, penalty = 'l2', class_weight =
                                           'auto')
  adareg = AdaBoostRegressor(logreg, n_estimators = 50, random_state = 33)  


  logreg.fit(titanic_X, traininglabels)
  adareg.fit(titanic_X, traininglabels)
  adaginitree.fit(titanic_X, traininglabels)
  print "Trying AdaBoost Logistic Regression:"
  #loo_cv(titanic_X, traininglabels, adareg)
  print "Trying AdaBoost Gini Tree:"
  #loo_cv(titanic_X, traininglabels, adaginitree)
  print "Trying AdaBoost Entropy Tree:"
  #loo_cv(titanic_X, traininglabels, adaentropytree)

  print metrics.confusion_matrix(adaginitree.predict(titanic_X_test),adareg.predict(
                        titanic_X_test))

  adaregpred = adareg.predict(titanic_X_test)
  adaginipred = adaginitree.predict(titanic_X_test)
  #adaentropypred = adaentropytree.predict(titanic_X_test)

  writeOutput(adaregpred, "Highadalogreg2")
  writeOutput(adaginipred, "Highadaginitree2")
  print adaginipred.shape, titanic_X_test.shape
  #writeOutput(adaentropypred, "Highadaentropytree")


trainingSet, testingSet, trainingLabels = prepDataSet(columnList)

X_train, X_test, Y_train, Y_test = train_test_split(trainingSet, trainingLabels, test_size=0.20, random_state=42)
#trainMediumModels(trainingSet, testingSet, trainingLabels)

rfclf = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None)
rfclf.fit(X_train, Y_train)

Y_score = rfclf.predict_proba(X_test)

fpr, tpr, _ = roc_curve(Y_test[:], Y_score[:, 1])
roc_auc = auc(fpr, tpr)
print fpr, tpr, roc_auc


def plotROC(fpr, tpr, roc_auc):
  """ PLots a ROC Curve"""
  plt.figure()
  plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], 'k--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic example')
  plt.legend(loc="lower right")
  plt.show()

  # Plot ROC curve
  plt.figure()
  plt.plot(fpr, tpr,
           label='micro-average ROC curve (area = {0:0.2f})'
                 ''.format(roc_auc))
  plt.plot(fpr, tpr, label='ROC curve of class {0} (area = {1:0.2f})'
                                     ''.format(1,roc_auc))

  plt.plot([0, 1], [0, 1], 'k--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Some extension of Receiver operating characteristic to multi-class')
  plt.legend(loc="lower right")
  plt.show()

plotROC(fpr, tpr, roc_auc)








