import cPickle
import json
from math import sqrt
from os.path import isfile

import numpy as np
import pandas as pd
import xgboost as xgb
from django.http import HttpResponse
from django.http import HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
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

@csrf_exempt
def linear(request):
    if request.method == 'POST':
        req = json.loads(request.body)
        filename = req['log']
        expected_filename = "Results/linearregression"+filename+'.csv'
        data = results(expected_filename)
        if data is not None:
            return HttpResponse(data.to_json(), content_type="application/json")
        train_data, test_data, original_test_data = prep_data(filename)
        lm = LinearRegression(fit_intercept=True)
        y = train_data['remainingTime']
        train_data = train_data.drop('remainingTime', 1)
        lm.fit(train_data, y)
        with open('predictionmodels/time/linearregression_'+filename+'.pkl', 'wb') as fid:
            cPickle.dump(lm, fid)

        original_test_data['prediction'] = lm.predict(test_data)

        original_test_data.to_csv("Results/linearregression" + filename + '.csv', sep=',', mode='w+', index=False)
        original_test_data['remainingTime'] = original_test_data['remainingTime'] / 3600
        original_test_data['prediction'] = original_test_data['prediction'] / 3600
        original_test_data = to_return_data(original_test_data)
        return HttpResponse(original_test_data.to_json(), content_type="application/json")
    else:
        filename = request.GET['log']
        expected_filename = "Results/linearregression"+filename+'.csv'
        data = results(expected_filename)
        if data is not None:
            return HttpResponse(data.to_json(), content_type="application/json")
        else:
            return HttpResponseBadRequest()

@csrf_exempt
def randomforestregression(request):
    if request.method == 'POST':
        req = json.loads(request.body)
        filename = req['log']
        expected_filename = "Results/randomforestregression"+filename+'.csv'
        data = results(expected_filename)
        if data is not None:
            return HttpResponse(data.to_json(), content_type="application/json")
        train_data, test_data, original_test_data = prep_data(filename)
        rf = RandomForestRegressor(n_estimators=50, n_jobs=8, verbose=1)
        y = train_data['remainingTime']
        train_data = train_data.drop('remainingTime', 1)
        rf.fit(train_data, y)
        with open('predictionmodels/time/randomforestregression'+filename+'.pkl', 'wb') as fid:
            cPickle.dump(rf, fid)

        original_test_data['prediction'] = rf.predict(test_data)

        original_test_data.to_csv("Results/randomforestregression"+filename+'.csv',sep=',',mode='w+', index=False)
        original_test_data['remainingTime'] = original_test_data['remainingTime'] / 3600
        original_test_data['prediction'] = original_test_data['prediction'] / 3600
        original_test_data = to_return_data(original_test_data)
        return HttpResponse(original_test_data.to_json(), content_type="application/json")
    else:
        filename = request.GET['log']
        expected_filename = "Results/randomforestregression"+filename+'.csv'
        data = results(expected_filename)
        if data is not None:
            return HttpResponse(data.to_json(), content_type="application/json")
        else:
            return HttpResponseBadRequest()

@csrf_exempt
def xgboost(request):
    if request.method == 'POST':
        req = json.loads(request.body)
        filename = req['log']
        expected_filename = "Results/xgboostregression"+filename+'.csv'
        data = results(expected_filename)
        if data is not None:
            return HttpResponse(data.to_json(), content_type="application/json")
        train_data, test_data, original_test_data = prep_data(filename)
        clf = xgb.XGBRegressor(n_estimators=2000, max_depth=10)
        y = train_data['remainingTime']
        train_data = train_data.drop('remainingTime', 1)
        clf.fit(train_data, y)
        with open('predictionmodels/time/xgboost'+filename+'.pkl', 'wb') as fid:
            cPickle.dump(clf, fid)

        original_test_data['prediction'] = clf.predict(test_data)

        original_test_data.to_csv("Results/xgboostregression"+filename+'.csv', sep=',', mode='w+', index=False)

        original_test_data['remainingTime'] = original_test_data['remainingTime'] / 3600
        original_test_data['prediction'] = original_test_data['prediction'] / 3600

        original_test_data = to_return_data(original_test_data)
        return HttpResponse(original_test_data.to_json(), content_type="application/json")
    else:
        filename = request.GET['log']
        expected_filename = "Results/xgboostregression"+filename+'.csv'
        data = results(expected_filename)
        if data is not None:
            return HttpResponse(data.to_json(), content_type="application/json")
        else:
            return HttpResponseBadRequest()

def prep_data(filename):
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

def results(expected_filename):
    if isfile(expected_filename):
        data = pd.read_csv(expected_filename)
        data['remainingTime'] = data['remainingTime']/3600
        data['prediction'] = data['prediction'] / 3600
        data = to_return_data(data)
        return data
    return None

def lineargeneral(request):
    filename = request.GET['log']
    expectedFile = "Results/linearregression"+filename+'.csv'
    results = general(expectedFile)
    return HttpResponse(json.dumps(results), content_type="application/json")

def randomforestregressiongeneral(request):
    filename = request.GET['log']
    expectedFile = "Results/randomforestregression"+filename+'.csv'
    results = general(expectedFile)
    return HttpResponse(json.dumps(results), content_type="application/json")

def xgboostgeneral(request):
    filename = request.GET['log']
    expectedFile = "Results/xgboostregression"+filename+'.csv'
    results = general(expectedFile)
    return HttpResponse(json.dumps(results), content_type="application/json")

def general(expectedFile):
    results = {}
    if isfile(expectedFile):
        data = pd.read_csv(expectedFile)
        data['remainingTime'] = data['remainingTime']/3600
        data['prediction'] = data['prediction'] / 3600

        results['RMSE'] = root_mean_square_error_calculation(data, 'remainingTime', 'prediction')
        results['MAE'] = mean_absolute_error_calculation(data, 'remainingTime', 'prediction')
    return results

def linearevaluation(request):
    filename = request.GET['log']
    expectedFile = "Results/linearregression"+filename+'.csv'
    results = evaluation(expectedFile)
    return HttpResponse(json.dumps(results), content_type="application/json")

def randomforestregressionevaluation(request):
    filename = request.GET['log']
    expectedFile = "Results/randomforestregression"+filename+'.csv'
    results = evaluation(expectedFile)
    return HttpResponse(json.dumps(results), content_type="application/json")

def xgboostevaluation(request):
    filename = request.GET['log']
    expectedFile = "Results/xgboostregression"+filename+'.csv'
    results = evaluation(expectedFile)
    return HttpResponse(json.dumps(results), content_type="application/json")

def evaluation(expectedFile):
    data = pd.read_csv(expectedFile)
    if not data.empty:
        data['remainingTime'] = data['remainingTime']/3600
        data['prediction'] = data['prediction'] / 3600

        data = data[data.remainingTime != data.prediction]

        range = {}
        range_list = {}
        remaining_time = data['remainingTime']
        divider = 20

        range_difference = np.amax(remaining_time) / divider
        x = 0
        counter = 20
        while x < 20:
            counter -= 1
            range_start = (counter+1)*range_difference -1
            range_end = counter * range_difference
            range_string = "%s - %s" %(range_start, range_end)
            #filter data
            range_data = data[data['remainingTime'] <= range_start ]
            range_data = range_data[range_data['remainingTime'] >= range_end]

            if len(range_data) == 0:
                x += 1
                continue

            results = {}
            results['RMSE'] = root_mean_square_error_calculation(range_data, 'remainingTime', 'prediction')
            results['MAE'] = mean_absolute_error_calculation(range_data, 'remainingTime', 'prediction')

            range[x] = results
            range_list[x] = range_string
            x += 1
        results = {}
        results['data'] = range
        results['intervals'] = range_list
        return results
    return {}


def root_mean_square_error_calculation(df, field1, field2):
    return sqrt(mean_squared_error(df[field1], df[field2]))

def mean_absolute_error_calculation(df, field1, field2):
    return mean_absolute_error(df[field1], df[field2])