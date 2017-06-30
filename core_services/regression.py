import cPickle
import os.path
from os.path import isfile
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans



def linear(fileName, prefix, encoding, cluster):

    if isfile('core_results/' + fileName + '/' + str(prefix) + '/' + 'linearregression_' + encoding + '_'  + cluster + '_clustering.csv'):
        return None

    train_data, test_data, original_test_data = prep_data(
        fileName, prefix, encoding)
    
    make_dir('core_predictionmodels/' + fileName + '/' + str(prefix))
    make_dir('core_results/' + fileName + '/' + str(prefix))

    if cluster != "None":
        estimator = KMeans(n_clusters=3)
        estimator.fit(train_data)
        cluster_lists = {i: train_data.iloc[np.where(estimator.labels_ == i)[0]] for i in range(estimator.n_clusters)}
        writeHeader = True
        for cluster_list in cluster_lists:
            clusterd_data = cluster_lists[cluster_list]
            print clusterd_data
            original_cluster_data = cluster_lists[cluster_list]
            lm = Lasso(fit_intercept=True, warm_start=True)
            y = clusterd_data['remainingTime']
            clusterd_data = clusterd_data.drop('remainingTime', 1)
            lm.fit(clusterd_data, y)
            original_cluster_data['prediction'] = lm.predict(clusterd_data)
            if writeHeader is True:
                original_cluster_data.to_csv('core_results/' + fileName + '/' + str(prefix) + '/' +
                              'linearregression_' + encoding + '_'  + cluster + '_clustering.csv', sep=',',header=True, mode='a', index=False)
                writeHeader = False

            else:
                original_cluster_data.to_csv('core_results/' + fileName + '/' + str(prefix) + '/' +
                              'linearregression_' + encoding + '_'  + cluster + '_clustering.csv', sep=',',header=False, mode='a', index=False)
    else:
        lm = Lasso(fit_intercept=True, warm_start=True)
        y = train_data['remainingTime']
        train_data = train_data.drop('remainingTime', 1)
        lm.fit(train_data, y)

        with open('core_predictionmodels/' + fileName + '/' + str(prefix) + '/' + 'linearregression_' + encoding + '.pkl', 'wb') as fid:
            cPickle.dump(lm, fid)

        original_test_data['prediction'] = lm.predict(test_data)
        original_test_data.to_csv('core_results/' + fileName + '/' + str(prefix) + '/' +
                                  'linearregression_' + encoding + '_'  + cluster + '_clustering.csv', sep=',', mode='w+', index=False)


def randomforestregression(fileName, prefix, encoding, cluster):
    if isfile('core_results/' + fileName + '/' + str(prefix) + '/' + 'randomforest_' + encoding + '.csv'):
        return None

    train_data, test_data, original_test_data = prep_data(
        fileName, prefix, encoding)
    rf = RandomForestRegressor(n_estimators=50, n_jobs=8, verbose=1)
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
    clf = xgb.XGBRegressor(n_estimators=2000, max_depth=10)
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

    train_data = train_data.drop('Id', 1)
    original_test_data = test_data
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
