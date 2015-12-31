import pandas as pd
from textblob import TextBlob
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import sys
import xgboost as xgb
import re
from nltk.corpus import wordnet

#reload(sys)  
#sys.setdefaultencoding('utf8')
#italian         7838
#mexican         6438
#southern_us     4320
#indian          3003
#chinese         2673
#french          2646
#cajun_creole    1546
#thai            1539
#japanese        1423
#greek           1175
#spanish          989
#korean           830
#vietnamese       825
#moroccan         821
#british          804
#filipino         755
#irish            667
#jamaican         526
#russian          489
#brazilian        467
#Name: cuisine, dtype: int64

class RecipeLearner(object):
    k_folds = 4
    n_times = 1
    classifiers = [ 'xgboost', 'rf' ]# , ''
    
    def __init__(self, y = None, given_trainset = None, final_testset = None, test_index = None, approach = None):
        # set seed to reproduce results
        np.random.seed(786)   
        
        self.y_original = y
        self.target = y.name
        self.given_trainset = given_trainset
        self.final_testset = final_testset
        self.approach = approach
#        self.file_multi_train = 'ensemble_train_multi_' + approach + '.csv'
#        self.file_multi_test = 'ensemble_test_multi_' + approach + '.csv'
        self.file_ensemble_train = 'ensemble_train_' + approach + '.csv'
        self.file_ensemble_test = 'ensemble_test_' + approach + '.csv'
        self.file_ensemble_final_train = 'ensemble_train.csv'
        self.file_ensemble_final_test = 'ensemble_test.csv'
        
        cuisine_counts = self.y_original.value_counts()
        self.categories = list(cuisine_counts.index)
        
        self.ensemble_test = pd.DataFrame(index = test_index) # 
        
    def dictMap(self, listOfMajors):
        mapped_dict = {}
        for i, major in enumerate(reversed(listOfMajors)):
            mapped_dict[major] = i
        return mapped_dict 
    
    def blendTarget(self):
        
        self.ensemble_train = pd.read_csv(self.file_ensemble_final_train, index_col = 'id')
        self.ensemble_test = pd.read_csv(self.file_ensemble_final_test, index_col = 'id')
        
        Y = self.ensemble_train['cuisine'].values
        label_encoder = LabelEncoder()
        label_encoder.fit(Y)
        cols_x = []
        for col in self.ensemble_train.columns:
            if col == 'cuisine':
                continue
            cols_x.append(col)
            self.ensemble_train[col] = label_encoder.transform(self.ensemble_train[col].values)
            self.ensemble_test[col] = label_encoder.transform(self.ensemble_test[col].values)
        
        self.ensemble_train.drop('cuisine', axis=1, inplace=True)  
        y = label_encoder.transform(Y)
        X = self.ensemble_train.values
        X_test = self.ensemble_test.values
        
        blender = GradientBoostingClassifier()
        
        skf = StratifiedKFold(Y, 4, shuffle=True)
        scores = []
        for train_idx, valid_idx in skf:
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]
            blender_fit = blender.fit( X_train, y_train )
            y_pred = blender_fit.predict(X_valid)
#            print 'fold accuracy of blending = ' + str(accuracy_score(y_valid, y_pred))
            scores.append(accuracy_score(y_valid, y_pred))
        
        print 'overall accuracy of blending = ' + str( np.mean(scores) )
        blender_fit = blender.fit( X, y )
        p = blender_fit.predict(X_test)
        self.ensemble_test['cuisine'] = label_encoder.inverse_transform(p)
        self.ensemble_test.drop(cols_x, axis = 1, inplace=True)
        self.ensemble_test.to_csv('submission_20_2.csv', index = True)
        return self.ensemble_test    
    
    def analyzeTarget(self):
        """
        Here we analyze the nature of the target especially how many classes present...
        If > 2 then should we design multiple binary classifiers or just single multiclass classifier?
        """
        valid_split = self.stratifiedShuffleSplit()
            
        if valid_split == False:
            return        
        
#        multi_train, multi_test = self.validateMultipleBinary()
        single_train, single_test = self.validateSingleMulti()
        
        self.ensemble_train.to_csv(self.file_ensemble_train, index = True)
        self.ensemble_test.to_csv(self.file_ensemble_test, index = True)
        
        Y = self.ensemble_train['cuisine'].values
        label_encoder = LabelEncoder()
        label_encoder.fit(Y)
        cols_x = []
        for col in self.ensemble_train.columns:
            if col == 'cuisine':
                continue
            cols_x.append(col)
            self.ensemble_train[col] = label_encoder.transform(self.ensemble_train[col].values)
            self.ensemble_test[col] = label_encoder.transform(self.ensemble_test[col].values)
        
        self.ensemble_train.drop('cuisine', axis=1, inplace=True)  
        y = label_encoder.transform(Y)
        X = self.ensemble_train.values
        X_test = self.ensemble_test.values
        
        blender = GradientBoostingClassifier()
        
        skf = StratifiedKFold(Y, 4, shuffle=True)
        scores = []
        for train_idx, valid_idx in skf:
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]
            blender_fit = blender.fit( X_train, y_train )
            y_pred = blender_fit.predict(X_valid)
#            print 'fold accuracy of blending = ' + str(accuracy_score(y_valid, y_pred))
            scores.append(accuracy_score(y_valid, y_pred))
        
        print 'overall accuracy of blending = ' + str( np.mean(scores) )
        blender_fit = blender.fit( X, y )
        p = blender_fit.predict(X_test)
        self.ensemble_test['cuisine'] = label_encoder.inverse_transform(p)
        self.ensemble_test.drop(cols_x, axis = 1, inplace=True)
        return self.ensemble_test
        
    def trainByMultipleBinary(self):
            
        for clf in self.classifiers:
            print
            print 'training ' + clf
            validation = pd.DataFrame(index = self.ensemble_train.index, columns = self.categories)
            testing = pd.DataFrame(index = self.ensemble_test.index, columns = self.categories)
            
            for cuisine in self.categories:
                print cuisine
                self.target = cuisine
                self.y = self.y_original.copy(deep = True)
                self.y[ self.y != cuisine ] = 0
                self.y[ self.y == cuisine ] = 1
                self.y = self.y.astype(int)
                
                y = self.y.values        
                self.y_train, self.y_test = y[self.train_index], y[self.test_index] 
                    
                valid_training = self.trainClassifiers(clf, False, cuisine)
            
                if valid_training:
                    validation[cuisine] = self.train(self.x_train, self.y_train, self.x_test, False)
                    testing[cuisine] = self.train(self.given_trainset, self.y, self.final_testset, False)
            
            validation['cuisine'] = validation.idxmax(axis=1)
            self.ensemble_train['multi_' + clf] = validation['cuisine'].copy(deep = True)
            
            testing['cuisine'] = testing.idxmax(axis=1)
            self.ensemble_test['multi_' + clf] = testing['cuisine'] #.copy(deep = True)
        
    def validateMultipleBinary(self):
        print 'Validating by multiple binary classifiers approach...'
        try:
            self.ensemble_train = pd.read_csv(self.file_ensemble_train, index_col = 'id')
            self.ensemble_test = pd.read_csv(self.file_ensemble_test, index_col = 'id')
        except:
            print "the validation set and test sets are not yet prepared..."
            self.trainByMultipleBinary()
            
#            ensemble_train = self.ensemble_train
#            ensemble_test = self.ensemble_test
        
        for clf in self.classifiers:
            # confusion matrix
            y = self.ensemble_train['cuisine']
            y_pred = self.ensemble_train['multi_' + clf]
        
#            print '\nFinal confusion matrix:'
#            print confusion_matrix(y, y_pred)
            accuracy = round(accuracy_score(y, y_pred), 4)
            print '\naccuracy of multiple binary classifiers approach by ' + clf + ' = ' + str(accuracy)
            print 'Classification report:'
            print classification_report(y, y_pred)
        
        return self.ensemble_train, self.ensemble_test
    
    def trainBySingleMulti(self):
            
        print 'training rf'
#        self.validation = pd.DataFrame(index = self.ensemble_train.index, columns = self.categories)
#        self.testing = pd.DataFrame(index = self.ensemble_test.index, columns = self.categories)
        
#            self.mapped_cuisines = self.dictMap(self.categories)
#            self.y_encoded = self.y_original.map( self.mapped_cuisines ).astype(int)        
#            self.reverse_map_cuisines = dict((v, k) for k, v in self.mapped_cuisines.iteritems())            
        
        print 'cuisine'
        self.target = 'cuisine'
        self.y = self.y_original.copy(deep = True)
        
        y = self.y.values        
        self.y_train, self.y_test = y[self.train_index], y[self.test_index] 
            
        valid_training = self.trainClassifiers('rf', True, self.target)
    
        if valid_training:
            self.ensemble_train['single_rf'] = self.train(self.x_train, self.y_train, self.x_test, True)
            self.ensemble_test['single_rf'] = self.train(self.given_trainset, self.y, self.final_testset, True)
    
    def trainBySingleMultiGBR(self):
            
        print 'training gbr'
        
        print 'cuisine'
        self.target = 'cuisine'
        self.y = self.y_original.copy(deep = True)
        
        y = self.y.values        
        self.y_train, self.y_test = y[self.train_index], y[self.test_index] 
            
        valid_training = self.trainClassifiers('gbr', True, self.target)
    
        if valid_training:
            self.ensemble_train['single_gbr'] = self.train(self.x_train, self.y_train, self.x_test, True)
            self.ensemble_test['single_gbr'] = self.train(self.given_trainset, self.y, self.final_testset, True)
   
    def validateSingleMulti(self):
        print 'Validating by single multiclass classifier approach...'
        try:
            self.ensemble_train = pd.read_csv(self.file_ensemble_train, index_col = 'id')
            self.ensemble_test = pd.read_csv(self.file_ensemble_test, index_col = 'id')
        except:
            print "the validation set and test sets are not yet prepared..."
            self.trainBySingleMulti()
            self.trainBySingleMultiGBR()
            
        # confusion matrix
        y = self.ensemble_train['cuisine']
        y_pred = self.ensemble_train['single_rf']
        y_pred_gbr = self.ensemble_train['single_gbr']
    
        accuracy = round(accuracy_score(y, y_pred), 4)
        print '\naccuracy of single multiclass classifier approach by rf = ' + str(accuracy)
        print 'Classification report:'
        print classification_report(y, y_pred)
        
        accuracy = round(accuracy_score(y, y_pred_gbr), 4)
        print '\naccuracy of single multiclass classifier approach by gbr = ' + str(accuracy)
        print 'Classification report:'
        print classification_report(y, y_pred_gbr)
        
        return self.ensemble_train, self.ensemble_test
        
    def stratifiedShuffleSplit(self):
        sss = StratifiedShuffleSplit(self.y_original.values, 1, test_size=0.20)
        self.train_index, self.test_index = list(sss)[0]
        self.x_train, self.x_test = self.given_trainset[self.train_index], self.given_trainset[self.test_index]
        self.ensemble_train = pd.DataFrame(index = self.test_index)
        self.ensemble_train.index.name = 'id'
        self.ensemble_train['cuisine'] = self.y_original.values[self.test_index]
        return True
    
    def stratifiedKfoldCV(self, classifier, multiclass):
        """
        'classifier : 'rf' | 'xgboost' | 'all'
        """
        # setup parameters for xgboost
        param = {}
        
        if multiclass == False:
            param['objective'] = 'binary:logistic'
            param['eval_metric'] = 'error'
        else:
            param['objective'] = 'multi:softmax'
            param['eval_metric'] = 'merror'
            param['num_class'] = len(self.categories)
            
        # scale weight of positive examples
        param['eta'] = 0.1
        param['max_depth'] = 6
        param['silent'] = 1
        param['subsample'] = 0.7
        param['colsample_bytree'] = 0.8
#        param['max_delta_step'] = 1

        fold = 1
        skf = StratifiedKFold(self.y_train, n_folds=self.k_folds, shuffle=True) 
        
        y = self.y_train      
        
        self.n_rounds = []
        for validation_index, train_index in skf:
            x_train, x_validate = self.x_train[train_index], self.x_train[validation_index]
            y_train, y_validate = y[train_index], y[validation_index] 
            
            if classifier == 'gbr' or classifier == 'all':
                gbr = GradientBoostingClassifier()
                
                booster = gbr.fit( x_train, y_train )
                y_prediction = booster.predict(x_validate)
                self.predictions.loc[self.predictions.iloc[validation_index].index, 'gradient boosting classifier'] = y_prediction
            
            if classifier == 'rf' or classifier == 'all':
                # Initialize a Random Forest classifier
                rf = RandomForestClassifier(n_estimators=100, criterion='entropy') 
                
                # Fit the forest to the training set, using the bag of words as 
                # features and the sentiment labels as the response variable
                forest = rf.fit( x_train, y_train )
                    
                y_prediction = forest.predict(x_validate)
    #            print 'fold : ' + str(fold) + ' accuracy rf = ' + str(round(accuracy_score(y_validate, y_prediction), 4))
                
                # save the probabilities to predictions        
                self.predictions.loc[self.predictions.iloc[validation_index].index, 'random forest classifier'] = y_prediction        
            
            if classifier == 'xgboost' or classifier == 'all':
                # xgboost classifier
                dtrain = xgb.DMatrix(x_train, label=y_train )
                dval = xgb.DMatrix(x_validate, label=y_validate ) # 
                watchlist = [  (dval, 'test'), (dtrain,'train') ]
                num_round = 500
                clf = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=10, verbose_eval=False) # 
                
                y_prediction = clf.predict(dval)
                y_prediction[y_prediction >= 0.5] = 1
                y_prediction[y_prediction < 0.5] = 0
    #            print 'fold : ' + str(fold) + ' accuracy xgb = ' + str(round(accuracy_score(y_validate, y_prediction), 4))
                
                # save the probabilities to predictions        
                self.predictions.loc[self.predictions.iloc[validation_index].index, 'xgboost classifier'] = y_prediction                
                self.n_rounds.append(clf.best_iteration)
            fold += 1
        
            
    def train(self, train_set, y, test_set, multiclass):
        # setup parameters for xgboost
        param = {}
        
        if multiclass == False:
            param['objective'] = 'binary:logistic'
            param['eval_metric'] = 'error'
        else:
            param['objective'] = 'multi:softmax'
            param['eval_metric'] = 'merror'
            param['num_class'] = len(self.categories)
            
        # scale weight of positive examples
        param['eta'] = 0.1
        param['max_depth'] = 6
        param['silent'] = 1
        param['subsample'] = 0.7
        param['colsample_bytree'] = 0.8
#        param['max_delta_step'] = 1
        
        if self.best_clf == 'gbr':
            booster = GradientBoostingClassifier()
            booster_whole = booster.fit( train_set, y )
            
            if multiclass:
                p = booster_whole.predict(test_set)
            else:
                p = booster_whole.predict_proba(test_set)[:,1]
        elif self.best_clf == 'rf':
            rf = RandomForestClassifier(n_estimators=100, criterion='entropy') # , max_features=0.5
            
            # Fit the forest to the training set, using the bag of words as 
            # features and the sentiment labels as the response variable
            forest_whole = rf.fit( train_set, y )
            
            if multiclass:
                p = forest_whole.predict(test_set)
            else:
                p = forest_whole.predict_proba(test_set)[:,1]
        else:
            avg_rounds = int( np.mean(self.n_rounds) )
            dx = xgb.DMatrix( train_set, label=y )
            dy = xgb.DMatrix(test_set)
            xgb_whole = xgb.train(param, dx, avg_rounds)
            p = xgb_whole.predict(dy)
            
        return p

    def printKeyWords(self, clf, N = 5):
        pass
    
    def saveClassifier(self, filename):
        #file = open(filename,'w')
        file = open(filename,'wb')
        pickle.dump(self.clf,file)
        file.close()

    def determineBestCLassifier(self, clf):
        # confusion matrix
        y = self.predictions[self.target]
        self.best_clf = clf

        if clf == 'gbr' or clf == 'all':
            y_pred_gbr = self.predictions['gradient boosting classifier']
            
            print 'Gradient Boosting confusion matrix:'
            with pd.option_context('display.max_rows', 999, 'display.max_columns', 11, 'display.precision', 2, 'display.width', 1000):
                print confusion_matrix(y, y_pred_gbr)
            gbr_accuracy = round(accuracy_score(y, y_pred_gbr), 4)
            print 'Gradient Boosting accuracy = ' + str(gbr_accuracy)
        
        if clf == 'rf' or clf == 'all':
            y_pred_rf = self.predictions['random forest classifier']
            
            print 'Random Forest confusion matrix:'
            with pd.option_context('display.max_rows', 999, 'display.max_columns', 11, 'display.precision', 2, 'display.width', 1000):
                print confusion_matrix(y, y_pred_rf)
            rf_accuracy = round(accuracy_score(y, y_pred_rf), 4)
            print 'Random Forest accuracy = ' + str(rf_accuracy)
        
        if clf == 'xgboost' or clf == 'all':
            y_pred_xgb = self.predictions['xgboost classifier']
    
            print 'XGBoost confusion matrix:'
            with pd.option_context('display.max_rows', 999, 'display.max_columns', 11, 'display.precision', 2, 'display.width', 1000):
                print confusion_matrix(y, y_pred_xgb)
            xgb_accuracy = round(accuracy_score(y, y_pred_xgb), 4)
            print 'XGBoost accuracy = ' + str(xgb_accuracy)
        
        if clf == 'all':
            if rf_accuracy > xgb_accuracy:
                self.best_clf = 'rf'
                print 'Best classifier is RandomForestClassifier'        
            else:
                self.best_clf = 'xgboost'
                print 'Best classifier is XGBoost Classifier'  

    def trainClassifiers(self, clf, multiclass, target):
        y = self.y_train 
        self.predictions = pd.DataFrame(y, columns=[target])
        self.predictions['random forest classifier'] = None
        self.predictions['xgboost classifier'] = None        
        self.predictions['gradient boosting classifier'] = None     
        
#        for n in range(self.n_times):
        self.stratifiedKfoldCV(clf, multiclass)
        self.determineBestCLassifier(clf)
        
        return True
        
# The Feedback Processor
class RecipeProcessor:
    recipes = pd.DataFrame()
    target = 'cuisine'
#    targets = ['italian', 'mexican', 'southern_us', 'indian', 'chinese', 'french', 'cajun_creole', 'thai', 'japanese',
#               'greek', 'spanish', 'korean', 'vietnamese', 'moroccan', 'british', 'filipino', 'irish', 'jamaican', 'russian',
#               'brazilian']
    features = { 'text' : 'ingredients', 'structured' : ['no_of_ingredients'] } # 
    
    def __init__(self, trainingfilename, testfilename, index_col):
        self.trainingfilename = trainingfilename
        self.testfilename = testfilename
        self.index_col = index_col
        
    def tokenize(self, text):
        REGEX = re.compile(r":\s*")
        return [tok.strip().lower() for tok in REGEX.split(text)]
        
    def readTrainingData(self):
        self.recipes = pd.read_csv(self.trainingfilename, index_col = self.index_col, encoding = 'utf8')
        
    def readTestData(self):
        self.testing = pd.read_csv(self.testfilename, index_col = self.index_col, encoding = 'utf8')
    
    def preprocessing(self):
#        text_feature = self.features['text']
#        text_feature_train = self.recipes[ text_feature ]
#        text_feature_test = self.testing[ text_feature ]
#        
#        corrections = { 
#                        'amarettus' : 'amaretto',
#                        'artichok' : 'artichoke',
#                        'asafetida' : 'asafoetida',
#                        'barbecued' : 'barbecue',
#                        'bas' : 'base',
#                        'bbq' : 'barbecue',
#                        'cardamom' : 'cardamon',
#                        'chee' : 'cheese',
#                        'chili' : 'chilli',
#                        'chilly' : 'chilli',
#                        'chily' : 'chilli'
#                        
#                    }        
#        
#        for correction in corrections:
#            print correction + ' => ' + corrections[correction]
#            self.recipes[ text_feature ] = text_feature_train.str.replace(correction, corrections[correction], case=False)
#            self.testing[ text_feature ] = text_feature_test.str.replace(correction, corrections[correction], case=False)
        
#        for index, row in self.recipes.iterrows():
#            b = TextBlob(row[text_feature])
#            print row['cuisine'] + ' :: ' + ' '.join( set ([w.detect_language() for w in b.words] ) )
#            self.recipes.loc[index, text_feature] = ' '.join( b.correct().words )
#            
#        for index, row in self.testing.iterrows():
#            b = TextBlob(row[text_feature])
#            self.testing.loc[index, text_feature] = ' '.join( b.correct().words )
        pass

    def packageDataForClassification(self, sample):
        packaged = []
        feature = self.features['text']
        for index, row in sample.iterrows():
            features_text = row[feature]
            packaged.append(features_text) 
            
        return packaged
    
    def transformCorpus(self, max_features = None):
        list_of_recipes_train = self.packageDataForClassification(self.recipes)
        list_of_recipes_test = self.packageDataForClassification(self.testing)
        
        recipes = list(list_of_recipes_train)
        recipes.extend(list_of_recipes_test)
        
        # Initialize the "CountVectorizer" object, which is scikit-learn's
        # bag of words tool.  
        vectorizer = CountVectorizer(analyzer = "word",   \
#                                     tokenizer = self.tokenize,    \
                                     preprocessor = None, \
                                     stop_words = None,   \
                                     binary = True,       \
                                     max_features = max_features
#                                     ngram_range = (1, 2)
                                     ) 

        fitted_vectorizer = vectorizer.fit(recipes)
        self.given_trainset = fitted_vectorizer.transform(list_of_recipes_train).toarray()
        self.final_testset = fitted_vectorizer.transform(list_of_recipes_test).toarray()
        
#        # Take a look at the words in the vocabulary
#        print 'Top Features:'
#        print fitted_vectorizer.get_feature_names()        
        
        if 'structured' in self.features:
#            print 'structured features present!'
            
            struc_feas = self.features['structured']
#            print struc_feas
            
            for struc_fea in struc_feas:
                train_f = self.recipes[struc_fea].values
                test_f = self.testing[struc_fea].values
                
                # add no of ingredients column
                self.given_trainset = np.column_stack((self.given_trainset, train_f))
                self.final_testset = np.column_stack((self.final_testset, test_f))
        
        # the shape of the array
        print self.given_trainset.shape

    def trainTarget(self, target = 'cuisine', extractKeywords = True):
        df = self.recipes.copy()
        
        y = df[target]    
        noOfRows = len(y)
        
        # print the classes / labels
        print '\n======= ' + target + ' (' + str(noOfRows) + ' rows) ======='
        print self.features

        print
        self.ml = RecipeLearner(y, self.given_trainset, self.final_testset, self.testing.index, 'testing_gbr')
        self.predictions = self.ml.analyzeTarget()
#        self.predictions.to_csv('submission_19_3.csv', index = True)
        
    def trainTargets(self):
        for target in self.targets:
            self.trainTarget(target)
    
    def ensemble(self):
        self.ml = RecipeLearner(y = self.recipes['cuisine'], approach = '')
        self.ml.blendTarget()
    
    def savePredictions(self):
#        self.testing.drop('ingredients', 1, inplace = True)
#        self.testing.drop('no_of_ingredients', 1, inplace = True)
#        self.testing.to_csv('predictions.csv', index = True)
        
#        self.testing['cuisine'] = self.testing.idxmax(axis=1)
        self.predictions.drop(self.ml.categories, 1, inplace = True)
#        print self.testing.columns
        self.predictions.to_csv('submission_17_1.csv', index = True)
        
############################################################################
def print_usage(filename):
	print 'Usage is:'
	print 'python', filename, '<trainingfile> target(optional)'
	
if __name__ == "__main__":
    
    processor = RecipeProcessor('train.csv', 'test.csv', 'id')
    processor.readTrainingData()      
#    processor.readTestData()
##    processor.preprocessing()
#    processor.transformCorpus()
#    processor.trainTarget(processor.target)
    processor.ensemble()
    
#    b = TextBlob('brisket')    
#    print b.detect_language()
#    synsets = wordnet.synsets( b ) 
#    # Print the information
#    
#    for synset in synsets:
#        lex_type = synset.lexname()
#        print "-" * 10
#        print "Name:", synset.name()
#        print "Lexical Type:", synset.lexname()
#        print "Lemmas:", synset.lemma_names()
#        print "Definition:", synset.definition()
#        for example in synset.examples():
#            print "Example:", example
    
#    first_ingredient = processor.recipes.ix[1]['ingredients']
#    ingres = first_ingredient.split(':')
#    for ingre in ingres:
#        blb = TextBlob(ingre)
##        print ingre
##        print blb.words.singularize()
#        blb = TextBlob(' '.join(blb.words.singularize()))
#        synsets = wordnet.synsets( blb )
#        print blb
#        multiple = False
#        if len(synsets) > 1:
#            multiple = True
##        print
#        # Print the information
#        for synset in synsets:
#            lex_type = synset.lexname()
#            if (multiple == True and lex_type == 'noun.food') or (multiple == False and lex_type == 'noun.plant'):
#                print "-" * 10
#                print "Name:", synset.name()
#                print "Lexical Type:", synset.lexname()
#                print "Lemmas:", synset.lemma_names()
#                print "Definition:", synset.definition()
##            for example in synset.examples:
##                print "Example:", example
#        print
    
#    test = pd.read_json('test.json')
#    test['no_of_ingredients'] = 0
#    test['languages_used'] = None
#    for index, row in test.iterrows():
#        blob = TextBlob( ' '.join( row['ingredients'] ) )
#        ingre_singular = blob.words.singularize()
#        test.loc[index, 'ingredients'] = ' '.join( ingre_singular )
#        test.loc[index, 'no_of_ingredients'] = len( row['ingredients'] )
#        test.loc[index, 'languages_used'] = ' '.join( set ([w.detect_language() for w in ingre_singular if len(w) >= 3 ] ) )   
#    test.to_csv('test_language.csv', index = False, encoding='utf8')
#    
#    train = pd.read_json('train.json') # , orient = 'records'
#    train['no_of_ingredients'] = 0
#    train['languages_used'] = None
#    for index, row in train.iterrows():
#        blob = TextBlob( ' '.join( row['ingredients'] ) )
#        ingre_singular = blob.words.singularize()
#        train.loc[index, 'ingredients'] = ' '.join( ingre_singular )
#        train.loc[index, 'no_of_ingredients'] = len( row['ingredients'] )
#        train.loc[index, 'languages_used'] = ' '.join( set ([w.detect_language() for w in ingre_singular if len(w) >= 3 ] ) )     
#    train.to_csv('train_language.csv', index = False, encoding='utf8')
    
#    train = pd.read_json('train.json') # , orient = 'records'
#    train['no_of_ingredients'] = 0
#    for index, row in train.iterrows():
##        ingredients = [ingredient.decode('utf8').strip() for ingredient in row['ingredients']]
##        blob = TextBlob( ' '.join( row['ingredients'] ) )
##        ingre_singular = blob.words.singularize()
#        train.loc[index, 'ingredients'] = ':'.join( row['ingredients'] )
#        train.loc[index, 'no_of_ingredients'] = len( row['ingredients'] )
#    train.to_csv('train_whole.csv', index = False, encoding='utf8')