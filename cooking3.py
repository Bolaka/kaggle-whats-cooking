# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 13:39:40 2015

@author: abzooba
"""

import pandas as pd
#from textblob import TextBlob
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import sys
import xgboost as xgb

train = pd.read_csv('train.csv', index_col = 'id', encoding = 'utf8')
test = pd.read_csv('test.csv', index_col = 'id', encoding = 'utf8')

def packageDataForClassification(sample, feature):
    packaged = []
    for index, row in sample.iterrows():
        features_text = row[feature]
        packaged.append(features_text) 
        
    return packaged

# setup parameters for xgboost
param = {}
#param['booster'] = 'gblinear'
param['objective'] = 'multi:softmax'
param['eval_metric'] = 'error'
# scale weight of positive examples
param['eta'] = 0.5
param['max_depth'] = 7
param['silent'] = 1
#param['subsample'] = 0.5
#param['colsample_bytree'] = 0.8

k_folds = 10
target = 'italian'

train[target] = 0
train.loc[ train['cuisine'] == target, target ] = 1

y = train[target].values
feature = 'ingredients'
fold = 1
skf = StratifiedKFold(y, n_folds=k_folds, shuffle=True) 

list_of_recipes_train = packageDataForClassification(train, feature)
list_of_recipes_test = packageDataForClassification(test, feature)

recipes = list(list_of_recipes_train)
recipes.extend(list_of_recipes_test)

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 50) 
# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
fitted_vectorizer = vectorizer.fit(recipes)
data_features = fitted_vectorizer.transform(list_of_recipes_train).toarray()
test_features = fitted_vectorizer.transform(list_of_recipes_test).toarray()

# Take a look at the words in the vocabulary
print 'Top Features:'
print fitted_vectorizer.get_feature_names()

predictions = pd.DataFrame(train[target], columns=[target])
predictions['random forest classifier'] = None
predictions['xgboost classifier'] = None

for train_index, validation_index in skf:
    #train_index, validation_index = list(skf)[0] 
    x_train, x_validate = data_features[train_index], data_features[validation_index]
    y_train, y_validate = y[train_index], y[validation_index] 
    
    # Initialize a Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100) 
    
    # Fit the forest to the training set, using the bag of words as 
    # features and the sentiment labels as the response variable
    forest = rf.fit( x_train, y_train )
        
    y_prediction = forest.predict(x_validate)
    print 'fold : ' + str(fold) + ' accuracy rf = ' + str(round(accuracy_score(y_validate, y_prediction), 4))
    
    # save the probabilities to predictions        
    predictions.loc[predictions.iloc[validation_index].index, 'random forest classifier'] = y_prediction        
    
#    # xgboost classifier
#    dtrain = xgb.DMatrix(x_train, label=y_train )
#    dval = xgb.DMatrix(x_validate, label=y_validate ) # 
#    watchlist = [  (dval, 'test'), (dtrain,'train') ]
#    num_round = 500
#    clf = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=30, verbose_eval=False) # 
#    
#    y_prediction = clf.predict(dval)
#    print y_prediction
#    y_prediction[y_prediction >= 0.5] = 1
#    y_prediction[y_prediction < 0.5] = 0
#    print 'fold : ' + str(fold) + ' accuracy rf = ' + str(round(accuracy_score(y_validate, y_prediction), 4))
#    
##     save the probabilities to predictions        
#    predictions.loc[predictions.iloc[validation_index].index, 'xgboost classifier'] = y_prediction    
    
    fold += 1

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
forest_whole = rf.fit( data_features, y )
p = forest_whole.predict(test_features)
print p
#test['cuisine'] = p
#test.drop('ingredients', 1, inplace = True)
#test.to_csv('submission_rf.csv', index = True)