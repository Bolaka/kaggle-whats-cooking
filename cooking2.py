# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 13:39:40 2015

@author: abzooba
"""

import pandas as pd
from textblob import TextBlob
import numpy as np
from textblob.classifiers import NaiveBayesClassifier
from textblob.classifiers import DecisionTreeClassifier
from textblob.classifiers import MaxEntClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import sys

train = pd.read_csv('train.csv', index_col = 'id', encoding = 'utf8')
test = pd.read_csv('test.csv', index_col = 'id', encoding = 'utf8')

def packageDataForClassification(sample, feature, target, withIndex = False):
    packaged = []
    
    for index, row in sample.iterrows():
        features_text = row[feature]
        if withIndex:
            packaged.append((index, features_text, row[target]))
        else:
            print row[target] + ' : ' + features_text
            packaged.append((features_text, row[target])) 
    return packaged

k_folds = 5
target = 'cuisine'
y = train[target]
feature = 'ingredients'

fold = 1
skf = StratifiedKFold(y, n_folds=k_folds, shuffle=True) 

#for train_index, validation_index in skf:
validation_index, train_index = list(skf)[0]  
train_data = train.loc[ train.iloc[train_index].index ]
test_data = train.loc[ train.iloc[validation_index].index ]
#print train_data.cuisine.value_counts()
#print test_data.cuisine.value_counts()
training = packageDataForClassification(train_data, feature, target)
testing = packageDataForClassification(test_data, feature, target, True)
testing_ = packageDataForClassification(test_data, feature, target)
#print training
# Naive Bayes Classifier
naive_clf = NaiveBayesClassifier(training)
#naive_accuracy = naive_clf.accuracy(testing_)

## Decision Tree Classifier
#tree_clf = DecisionTreeClassifier(train)
#tree_accuracy = tree_clf.accuracy(test_)

#for index, feature_x, y in test:
#    self.predictions.loc[ index, 'naive bayes classifier' ] = naive_clf.classify(feature_x)
#    self.predictions.loc[ index, 'decision tree classifier' ] = tree_clf.classify(feature_x)            

#print 'fold : ' + str(fold) + ':: Naive Bayes = ' + str(round(naive_accuracy, 4)) + '; Decision Tree = ' + str(round(tree_accuracy, 4))
#    fold += 1