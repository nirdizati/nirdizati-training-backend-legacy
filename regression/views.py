import collections
import json
from os.path import isfile

import numpy as np
from django.http import HttpResponse
from django.http import HttpResponseBadRequest
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

import pandas as pd

from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def index(request):
    return HttpResponse()

def split_data(data):
    cases = data['id'].unique()
    import random
    random.shuffle(cases)

    cases_train_point = int(len(cases) * 0.8)

    train_cases = cases[:cases_train_point]

    ids = []
    for i in range(0, len(data)):
        ids.append(data['id'][i] in train_cases)

    train_data = data[ids]
    test_data = data[np.invert(ids)]
    return train_data, test_data

def linear(request):
    filename = request.GET['log']
    expected_filename = "Results/linearregression"+filename+'.csv'
    if isfile(expected_filename):
        data = pd.read_csv(expected_filename)
        data['remainingTime'] = data['remainingTime']/3600
        data['prediction'] = data['prediction'] / 3600
        data = to_return_data(data)
        return HttpResponse(data.to_json(), content_type="application/json")

    train_data, test_data, original_test_data = prep_data(request)
    lm = LinearRegression(fit_intercept=True)
    y = train_data['remainingTime']
    train_data = train_data.drop('remainingTime', 1)
    lm.fit(train_data, y)

    original_test_data['prediction'] = lm.predict(test_data)

    original_test_data.to_csv("Results/linearregression"+filename+'.csv',sep=',',mode='w+', index=False)
    original_test_data['remainingTime'] = original_test_data['remainingTime'] / 3600
    original_test_data['prediction'] = original_test_data['prediction'] / 3600
    original_test_data = to_return_data(original_test_data)
    return HttpResponse(original_test_data.to_json(), content_type="application/json")

def randomforestregression(request):
    filename = request.GET['log']
    expected_filename = "Results/randomforestregression"+filename+'.csv'
    if isfile(expected_filename):
        data = pd.read_csv(expected_filename)
        data['remainingTime'] = data['remainingTime']/3600
        data['prediction'] = data['prediction'] / 3600
        data = to_return_data(data)
        return HttpResponse(data.to_json(), content_type="application/json")

    train_data, test_data, original_test_data = prep_data(request)
    rf = RandomForestRegressor(n_estimators=50, n_jobs=8, verbose=1)
    y = train_data['remainingTime']
    train_data = train_data.drop('remainingTime', 1)
    rf.fit(train_data, y)

    original_test_data['prediction'] = rf.predict(test_data)

    original_test_data.to_csv("Results/randomforestregression"+filename+'.csv',sep=',',mode='w+', index=False)
    original_test_data['remainingTime'] = original_test_data['remainingTime'] / 3600
    original_test_data['prediction'] = original_test_data['prediction'] / 3600
    original_test_data = to_return_data(original_test_data)
    return HttpResponse(original_test_data.to_json(), content_type="application/json")

def xgboost(request):
    filename = request.GET['log']
    expected_filename = "Results/xgboostregression"+filename+'.csv'
    if isfile(expected_filename):
        data = pd.read_csv(expected_filename)
        data['remainingTime'] = data['remainingTime']/3600
        data['prediction'] = data['prediction'] / 3600
        data = to_return_data(data)
        return HttpResponse(data.to_json(), content_type="application/json")

    train_data, test_data, original_test_data = prep_data(request)
    clf = xgb.XGBRegressor(n_estimators=2000, max_depth=10)
    y = train_data['remainingTime']
    train_data = train_data.drop('remainingTime', 1)
    clf.fit(train_data, y)

    original_test_data['prediction'] = clf.predict(test_data)

    original_test_data.to_csv(expected_filename,sep=',',mode='w+', index=False)

    original_test_data['remainingTime'] = original_test_data['remainingTime'] / 3600
    original_test_data['prediction'] = original_test_data['prediction'] / 3600

    original_test_data = to_return_data(original_test_data)
    return HttpResponse(original_test_data.to_json(), content_type="application/json")

def prep_data(request):
    filename = request.GET['log']
    df = pd.read_csv(filepath_or_buffer="encodedfiles/indexbased_" + filename + '.csv', header=0)

    train_data, test_data = split_data(df)

    train_data = train_data.drop('id', 1)
    original_test_data = test_data
    test_data = test_data.drop('id', 1)

    test_data = test_data.drop('remainingTime', 1)

    return train_data, test_data, original_test_data

def to_return_data(data):
    new_data = pd.DataFrame(index=range(0, len(data)), columns=['id', 'remainingTime', 'prediction'])
    new_data['id'] = data['id']
    new_data['remainingTime'] = data['remainingTime']
    new_data['prediction'] = data['prediction']

    return new_data

def evaluation(request):
    return HttpResponse()

def general(request):
    return HttpResponse()