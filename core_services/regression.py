import cPickle
import os.path
from os.path import isfile
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error


def linear(fileName, prefix, encoding, cluster):

    if isfile('core_results/' + fileName + '/' + str(prefix) + '/' + 'linearregression_' + encoding + '.csv'):
        return None

    train_data, test_data, original_test_data = prep_data(
        fileName, prefix, encoding)
    lm = Lasso(fit_intercept=True, warm_start=True)
    y = train_data['remainingTime']
    train_data = train_data.drop('remainingTime', 1)
    lm.fit(train_data, y)

    make_dir('core_predictionmodels/' + fileName + '/' + str(prefix))

    with open('core_predictionmodels/' + fileName + '/' + str(prefix) + '/' + 'linearregression_' + encoding + '.pkl', 'wb') as fid:
        cPickle.dump(lm, fid)

    make_dir('core_results/' + fileName + '/' + str(prefix))
    original_test_data['prediction'] = lm.predict(test_data)
    original_test_data.to_csv('core_results/' + fileName + '/' + str(prefix) + '/' +
                              'linearregression_' + encoding + '.csv', sep=',', mode='w+', index=False)


def randomforestregression(fileName, prefix, encoding, cluster):
    if isfile('core_results/' + fileName + '/' + str(prefix) + '/' + 'randomforest' + encoding + '.csv'):
        return None

    train_data, test_data, original_test_data = prep_data(
        fileName, prefix, encoding)
    rf = RandomForestRegressor(n_estimators=50, n_jobs=8, verbose=1)
    y = train_data['remainingTime']
    train_data = train_data.drop('remainingTime', 1)
    rf.fit(train_data, y)

    make_dir('core_predictionmodels/' + fileName + '/' + str(prefix))

    make_dir('core_results/' + fileName + '/' + str(prefix))

    with open('core_predictionmodels/' + fileName + '/' + str(prefix) + '/' + 'randomforest' + encoding + '.pkl', 'wb') as fid:
        cPickle.dump(rf, fid)

    original_test_data['prediction'] = rf.predict(test_data)
    original_test_data.to_csv('core_results/' + fileName + '/' + str(prefix) +
                              '/' + 'randomforest' + encoding + '.csv', sep=',', mode='w+', index=False)


def xgboost(fileName, prefix, encoding, cluster):
    if isfile('core_results/' + fileName + '/' + str(prefix) + '/' + 'xgboost' + encoding + '.csv'):
        return None
    train_data, test_data, original_test_data = prep_data(
        fileName, prefix, encoding)
    clf = xgb.XGBRegressor(n_estimators=2000, max_depth=10)
    y = train_data['remainingTime']
    train_data = train_data.drop('remainingTime', 1)
    clf.fit(train_data, y)

    make_dir('core_predictionmodels/' + fileName + '/' + str(prefix))

    make_dir('core_results/' + fileName + '/' + str(prefix))

    with open('core_predictionmodels/' + fileName + '/' + str(prefix) + '/' + 'xgboost' + encoding + '.pkl', 'wb') as fid:
        cPickle.dump(clf, fid)

    original_test_data['prediction'] = clf.predict(test_data)
    original_test_data.to_csv('core_results/' + fileName + '/' + str(
        prefix) + '/' + 'xgboost' + encoding + '.csv', sep=',', mode='w+', index=False)


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
