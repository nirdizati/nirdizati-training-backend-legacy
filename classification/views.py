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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.cross_validation import train_test_split
import encoding

def read(request):
    if request.GET['encodingType'] == "fastslow":
        return encoding.fast_slow_encode(request)
    elif request.GET['encodingType'] == "ltl":
        return encoding.ltl_encode(request)
    else:
        return None

def index(request):
    return HttpResponse()

def split_data(data):
    data = data.sample(frac=1)

    cases_train_point = int(len(data) * 0.8)

    train_df, test_df = train_test_split(data, test_size=0.2, random_state=3)
    return train_df, test_df

def prep_data(filename):
    df = pd.read_csv(filepath_or_buffer="encodedfiles/indexbased_" + filename + '.csv', header=0)

    train_data, test_data = split_data(df)

    train_data = train_data.drop('id', 1)
    original_test_data = test_data
    test_data = test_data.drop('id', 1)

    test_data = test_data.drop('remainingTime', 1)
    train_data = train_data.drop('remainingTime', 1)
    test_data = test_data.drop('elapsedTime', 1)
    train_data = train_data.drop('elapsedTime', 1)

    return train_data, test_data, original_test_data

def calculate_results(prediction, actual):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    for i in range(0, len(actual)):
        if actual[i] == prediction[i] & actual[i] == True:
            true_positive += 1
        elif actual[i] != prediction[i] & actual[i] == True:
            false_positive += 1
        elif actual[i] != prediction[i] & actual[i] == False:
            false_negative += 1
        elif actual[i] == prediction[i] & actual[i] == False:
            true_negative += 1

    precision = float(true_positive) / (true_positive + false_positive)

    recall = float(true_positive) / (true_positive + false_negative)
    f1score = (2 * precision * recall) / (precision + recall)
    return f1score

def dt(request):
    if request.method == 'GET':
        df = read(request)
        train_data, test_data = split_data(df)
        clf = DecisionTreeClassifier()
        to_predict = 'label'

        y = train_data[to_predict]

        train_data = train_data.drop(to_predict, 1)

        test_data = test_data.reset_index(drop=True)
        actual = test_data[to_predict]
        test_data = test_data.drop(to_predict, 1)

        clf.fit(train_data, y)

        prediction = clf.predict(test_data)
        test_data["actual"] = actual
        test_data["predicted"] = prediction
        f1score = calculate_results(prediction, actual)

        return HttpResponse(json.dumps(results(f1score, "DT", test_data.values.tolist())), content_type="application/json")


def rf(request):
    if request.method == 'GET':
        df = read(request)
        train_data, test_data = split_data(df)
        clf = RandomForestClassifier()
        to_predict = 'label'

        y = train_data[to_predict]

        train_data = train_data.drop(to_predict, 1)

        test_data = test_data.reset_index(drop=True)
        actual = test_data[to_predict]
        test_data = test_data.drop(to_predict, 1)

        clf.fit(train_data, y)

        prediction = clf.predict(test_data)
        test_data["actual"] = actual
        test_data["predicted"] = prediction
        f1score = calculate_results(prediction, actual)

        return HttpResponse(json.dumps(results(f1score, "RF", test_data.values.tolist())), content_type="application/json")

def knn(request):
    if request.method == 'GET':
        df = read(request)
        train_data, test_data = split_data(df)
        clf = KNeighborsClassifier()
        to_predict = 'label'

        y = train_data[to_predict]

        train_data = train_data.drop(to_predict, 1)

        test_data = test_data.reset_index(drop=True)
        actual = test_data[to_predict]
        test_data = test_data.drop(to_predict, 1)

        clf.fit(train_data, y)

        prediction = clf.predict(test_data)
        test_data["actual"] = actual
        test_data["predicted"] = prediction
        f1score = calculate_results(prediction, actual)

        return HttpResponse(json.dumps(results(f1score, "KNN", test_data.values.tolist())), content_type="application/json")

def results(f1score, method, data):
    res = {}
    res["accuracy"] = f1score
    res["method"] = method
    res["results"] = data

    return res