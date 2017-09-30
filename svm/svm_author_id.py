#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn import svm
from sklearn.metrics import accuracy_score

#Reduce training datasets (reduces accuracy but decreases time spent)
#Now commented as c is optimized
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

#switched to rbf kernel over linear and optimized c param to 10000
#resulting in a more complex dicision boundry
clf = svm.SVC(C=10000,kernel="rbf")
clf.fit(features_train, labels_train)

t0 = time()
pred = clf.predict(features_test)
print "predict time:", round(time()-t0, 3), "s"

test  = pred[50]
print "predicted point: ", test

totalChrisEmails = 0

for dataPoint in pred:
    if dataPoint == 1:
        totalChrisEmails += 1

print "there are ", totalChrisEmails, " emails from Chris"


acc = accuracy_score(labels_test, pred)

print "Accuracy: ", acc

#########################################################
