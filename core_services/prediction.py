import cPickle
import os.path
from os.path import isfile
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier

from math import sqrt
import csv
from encoding import fast_slow_encode
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

from sklearn import metrics



def classifier(fileName, prefix, encoding, cluster, method, label, threshold):
    if isfile('core_results_class/' + fileName + '/' + str(prefix) + '/' + method + '_' + encoding + '_'  + cluster + '_clustering.csv'):
        return None
    
    if method == "KNN" :
        clf = KNeighborsClassifier()
    elif method == "RandomForest" : 
        clf = RandomForestClassifier()
    elif method == "DecisionTree":  
        clf = DecisionTreeClassifier()

    df = fast_slow_encode(fileName, prefix, encoding, label, threshold)
    
    train_data, test_data = split_class_data(df)
    to_predict = 'label'
    auc = 0
    make_dir('core_results_class/' + fileName + '/' + str(prefix) + '/' + label + '/' + str(threshold))

    if cluster != "None":
        estimator = KMeans(n_clusters=3)
        estimator.fit(train_data)
        orginal_cluster_lists = {i: test_data.iloc[np.where(estimator.predict(test_data) == i)[0]] for i in range(estimator.n_clusters)}    
        cluster_lists = {i: train_data.iloc[np.where(estimator.labels_ == i)[0]] for i in range(estimator.n_clusters)}

        writeHeader = True
        x = 0
        for cluster_list in cluster_lists:
            clusterd_train_data = cluster_lists[cluster_list]
            clusterd_test_data = orginal_cluster_lists[cluster_list]
            
            y = clusterd_train_data[to_predict]
            clusterd_train_data = clusterd_train_data.drop(to_predict, 1)

            clusterd_test_data = clusterd_test_data.reset_index(drop=True)
            actual = clusterd_test_data[to_predict]
            clusterd_test_data = clusterd_test_data.drop(to_predict, 1)
            print '__________________________________'
            #print cluster_lists[cluster_list]
            
            clf.fit(clusterd_train_data, y)
            prediction = clf.predict(clusterd_test_data)
            
            scores =  clf.predict_proba(clusterd_test_data)
            if '1)' in str(scores.shape):
                auc += 0
            else:
                try:
                    auc += metrics.roc_auc_score(actual,scores[:,1])
                    x += 1
                except: 
                    auc += 0
                
                        

            clusterd_test_data["actual"] = actual
            clusterd_test_data["predicted"] = prediction    
            
            if writeHeader is True:
                clusterd_test_data.to_csv('core_results_class/' + fileName + '/' + str(prefix) + '/' + label + '/' + str(threshold) + '/' + 
                              method + '_' + encoding + '_'  + cluster + '_clustering.csv', sep=',',header=True, mode='a', index=False)
                writeHeader = False

            else:
                clusterd_test_data.to_csv('core_results_class/' + fileName + '/' + str(prefix) + '/' + label  + '/' + str(threshold) + '/' + 
                              method + '_' + encoding + '_'  + cluster + '_clustering.csv', sep=',',header=False, mode='a', index=False)
        
        try:
            auc = float(auc) / x
        except: 
            auc = 0
    else:
        y = train_data[to_predict]

        train_data = train_data.drop(to_predict, 1)

        test_data = test_data.reset_index(drop=True)
        actual = test_data[to_predict]
        test_data = test_data.drop(to_predict, 1)

        clf.fit(train_data, y)

        prediction = clf.predict(test_data)
        scores =  clf.predict_proba(test_data)[:,1]
        print scores

        print len(scores)
        print len(prediction)
        test_data["actual"] = actual
        test_data["predicted"] = prediction
        #FPR,TPR,thresholds_unsorted=
        auc = metrics.roc_auc_score(actual,scores) # ,pos_label='True')
        #auc = metrics.auc(FPR, TPR)
        

        test_data.to_csv('core_results_class/' + fileName + '/' + str(prefix) + '/' + label  + '/' + str(threshold) + '/' + 
                              method + '_' + encoding + '_'  + cluster + '_clustering.csv', sep=',', mode='w+', index=False)
        print test_data.shape

   
    df = pd.read_csv(filepath_or_buffer='core_results_class/' + fileName + '/' + str(prefix)+ '/' + label + '/' + str(threshold) + '/' +  method + '_' + encoding + '_'  + cluster + '_clustering.csv', header=0, index_col=0)
    actual_ = df['actual'].values
    predicted_ =  df['predicted'].values
    
    f1score, acc= calculate_results(actual_, predicted_)


    
    #auc = metrics.roc_auc_score(test_data, scores)
    methodVal = method + '_' + encoding + '_'  + cluster + '_clustering'
    # results = {method: methodVal, rmse: rmse, mae: mae}
    # results.to_csv('core_results/blabal.csv')
    writeHeader = True
    if isfile('core_results_class/' + fileName + '/' + str(prefix) + '/' + label + '/' + str(threshold) + '/General.csv'):
            writeHeader = False
    with open('core_results_class/' + fileName + '/' + str(prefix) + '/' + label + '/' + str(threshold) + '/General.csv', 'a') as csvfile:
        fieldnames = ['Run', 'Fmeasure', 'ACC', 'AUC']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if writeHeader is True:
            writer.writeheader()

        writer.writerow({'Run': methodVal, 'Fmeasure': f1score, 'ACC': acc, 'AUC' : auc})

def regressior(fileName, prefix, encoding, cluster, method):

    if isfile('core_results_regg/' + fileName + '/' + str(prefix) + '/' + method + '_' + encoding + '_'  + cluster + '_clustering.csv'):
        return None
    if method == "linear" :
        regressor = Lasso(fit_intercept=True, warm_start=True)
    elif method == "xgboost" : 
        regressor = xgb.XGBRegressor(n_estimators=2000, max_depth=10)
    elif method == "randomforest":  
        regressor = RandomForestRegressor(n_estimators=50, n_jobs=8, verbose=1)

    train_data, test_data, original_test_data = prep_data(
        fileName, prefix, encoding)
    
    make_dir('core_predictionmodels/' + fileName + '/' + str(prefix))
    make_dir('core_results_regg/' + fileName + '/' + str(prefix))

    if cluster != "None":
        estimator = KMeans(n_clusters=3)
        estimator.fit(train_data)
        estimator2 = KMeans(n_clusters=3)
        estimator2.fit(test_data)
    
        orginal_cluster_lists = {i: original_test_data.iloc[np.where(estimator.predict(original_test_data.drop('Id', 1)) == i)[0]] for i in range(estimator2.n_clusters)}    
        cluster_lists = {i: train_data.iloc[np.where(estimator.labels_ == i)[0]] for i in range(estimator.n_clusters)}
        # print orginal_cluster_lists
        writeHeader = True
        for cluster_list in cluster_lists:
            clusterd_train_data = cluster_lists[cluster_list]
            
            clusterd_test_data = orginal_cluster_lists[cluster_list]
            orginal_test_clustered_data = orginal_cluster_lists[cluster_list]
            clusterd_test_data = clusterd_test_data.drop('Id', 1 )
            clusterd_test_data = clusterd_test_data.drop('remainingTime', 1 )
            
            y = clusterd_train_data['remainingTime']
            clusterd_train_data = clusterd_train_data.drop('remainingTime', 1)
            
            regressor.fit(clusterd_train_data, y)
            
            orginal_test_clustered_data['prediction'] = regressor.predict(clusterd_test_data)
            
            if writeHeader is True:
                orginal_test_clustered_data.to_csv('core_results_regg/' + fileName + '/' + str(prefix) + '/' +
                              method + '_' + encoding + '_'  + cluster + '_clustering.csv', sep=',',header=True, mode='a', index=False)
                writeHeader = False

            else:
                orginal_test_clustered_data.to_csv('core_results_regg/' + fileName + '/' + str(prefix) + '/' +
                              method + '_' + encoding + '_'  + cluster + '_clustering.csv', sep=',',header=False, mode='a', index=False)
    else:
        y = train_data['remainingTime']
        train_data = train_data.drop('remainingTime', 1)
        regressor.fit(train_data, y)

        with open('core_predictionmodels/' + fileName + '/' + str(prefix) + '/' + method + '_' + encoding + '.pkl', 'wb') as fid:
            cPickle.dump(regressor, fid)

        original_test_data['prediction'] = regressor.predict(test_data)
        original_test_data.to_csv('core_results_regg/' + fileName + '/' + str(prefix) + '/'
                                  + method + '_' + encoding + '_'  + cluster + '_clustering.csv', sep=',', mode='w+', index=False)


    df = pd.read_csv(filepath_or_buffer='core_results_regg/' + fileName + '/' + str(prefix) + '/' + method + '_' + encoding + '_'  + cluster + '_clustering.csv', header=0, index_col=0)
    df['remainingTime'] = df['remainingTime']/3600
    df['prediction'] = df['prediction'] / 3600
    rmse = sqrt(mean_squared_error(df['remainingTime'], df['prediction']))
    mae = mean_absolute_error(df['remainingTime'], df['prediction'])
    Rscore = metrics.r2_score(df['remainingTime'], df['prediction'])
    methodVal = method + '_' + encoding + '_'  + cluster + '_clustering'
    # results = {method: methodVal, rmse: rmse, mae: mae}
    # results.to_csv('core_results/blabal.csv')
    writeHeader = True
    if isfile('core_results_regg/' + fileName + '/' + str(prefix) + '/General.csv'):
            writeHeader = False
    with open('core_results_regg/' + fileName + '/' + str(prefix) + '/General.csv', 'a') as csvfile:
        fieldnames = ['Run', 'Rmse', 'Mae', 'Rscore']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if writeHeader is True:
            writer.writeheader()

        writer.writerow({'Run': methodVal, 'Rmse': rmse, 'Mae': mae, 'Rscore' : Rscore})
   
def randomforestregression(fileName, prefix, encoding, cluster):
    if isfile('core_results/' + fileName + '/' + str(prefix) + '/' + 'randomforest_' + encoding + '.csv'):
        return None

    train_data, test_data, original_test_data = prep_data(
        fileName, prefix, encoding)
    y = train_data['remainingTime']
    train_data = train_data.drop('remainingTime', 1)
    rf.fit(train_data, y)

    make_dir('core_predictionmodels/' + fileName + '/' + str(prefix))

    make_dir('core_results/' + fileName + '/' + str(prefix))

    with open('core_predictionmodels/' + fileName + '/' + str(prefix) + '/' + 'randomforest_' + encoding + '.pkl', 'wb') as fid:
        cPickle.dump(rf, fid)

    original_test_data['prediction'] = rf.predict(test_data)
    original_test_data.to_csv('core_results/' + fileName + '/' + str(prefix) +
                              '/' + 'randomForest_' + encoding + '.csv', sep=',', mode='w+', index=False)


def xgboost(fileName, prefix, encoding, cluster):
    if isfile('core_results/' + fileName + '/' + str(prefix) + '/' + 'xgboost_' + encoding + '.csv'):
        return None
    train_data, test_data, original_test_data = prep_data(
        fileName, prefix, encoding)
    y = train_data['remainingTime']
    train_data = train_data.drop('remainingTime', 1)
    clf.fit(train_data, y)

    make_dir('core_predictionmodels/' + fileName + '/' + str(prefix))

    make_dir('core_results/' + fileName + '/' + str(prefix))

    with open('core_predictionmodels/' + fileName + '/' + str(prefix) + '/' + 'xgboost_' + encoding + '.pkl', 'wb') as fid:
        cPickle.dump(clf, fid)

    original_test_data['prediction'] = clf.predict(test_data)
    original_test_data.to_csv('core_results/' + fileName + '/' + str(
        prefix) + '/' + 'xgboost_' + encoding + '.csv', sep=',', mode='w+', index=False)


def split_data(data):
    cases = data['Id'].unique()
    import random
    random.shuffle(cases)

    cases_train_point = int(len(cases) * 0.8)

    train_cases = cases[:cases_train_point]

    ids = []
    for i in range(0, len(data)):
        ids.append(data['Id'][i] in train_cases)

    train_data = data[ids]
    test_data = data[np.invert(ids)]
    return train_data, test_data


def prep_data(fileName, prefix, encoding):
    df = pd.read_csv(filepath_or_buffer='core_encodedFiles/' +
                     encoding + '_' + fileName + '_' + str(prefix) + '.csv', header=0)
    train_data, test_data = split_data(df)

    train_data = train_data.drop('duration', 1)
    test_data = test_data.drop('duration', 1)

    original_test_data = test_data

    train_data = train_data.drop('Id', 1)
    test_data = test_data.drop('Id', 1)


    test_data = test_data.drop('remainingTime', 1)

    return train_data, test_data, original_test_data

    # boolean_filename_prefix, frequency_filename_prefix, complex_index_filename_prefix, simple_index_filename_prefix, index_latest_payload


def make_dir(drpath):
    if not os.path.exists(drpath):
        try:
            os.makedirs(drpath)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def split_class_data(data):
    data = data.sample(frac=1)

    cases_train_point = int(len(data) * 0.8)

    train_df, test_df = train_test_split(data, test_size=0.2, random_state=3)
    return train_df, test_df

def calculate_results(prediction, actual):
    # true_positive = 0
    # false_positive = 0
    # false_negative = 0
    # true_negative = 0

    # for i in range(0, len(actual)):
    #     if actual[i] == True:
    #         if actual[i] == prediction[i]:
    #             true_positive += 1
    #         else:
    #             false_positive += 1
    #     else:
    #         if actual[i] == prediction[i]:
    #             true_negative += 1
    #         else:
    #             false_negative += 1
       
    #     # if actual[i] == prediction[i] and actual[i] == True:
    #     #     true_positive += 1
    #     # elif actual[i] != prediction[i] and actual[i] == True:
    #     #     false_positive += 1
    #     # elif actual[i] != prediction[i] and actual[i] == False:
    #     #     false_negative += 1
    #     # elif actual[i] == prediction[i] and actual[i] == False:
    #     #     true_negative += 1

    # print 'TP: ' + str(true_positive) + 'FP: ' + str(false_positive) + 'FN: ' + str(false_negative)
    # if true_positive == 0 and false_negative == 0 and false_positive == 0 :
    #     f1score = "uncomputable"
    # else:

    #     precision = float(true_positive) / (true_positive + false_positive)

    #     recall = float(true_positive) / (true_positive + false_negative)
    #     f1score = (2 * precision * recall) / (precision + recall)
    # acc = float(true_positive + true_negative) / (true_positive + true_negative + false_negative + false_positive )
    # # TPR = float(true_positive) / (true_positive + false_negative)
    # # FPR = float(false_positive) / (false_positive + true_negative)
    # # auc = metrics.auc(FPR, TPR)

    # # fpr_unsorted,tpr_unsorted,thresholds_unsorted=metrics.roc_curve(actual,prediction,pos_label='False')
    # # auc = metrics.roc_auc_score(actual, prediction, )
    
    f1score = metrics.f1_score(actual, prediction)
    acc = metrics.accuracy_score(actual, prediction)
    return f1score, acc