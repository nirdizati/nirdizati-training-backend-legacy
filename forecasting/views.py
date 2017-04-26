from django.shortcuts import render
from django.http import HttpResponse

from univariate_forecasting import UnivariateForecasting

import json
import collections


import numpy as np
from os import listdir
from os.path import isfile, join

def index(request):
    return HttpResponse()

def workload(request):
    filename = request.GET['log']
    expectedFile = "data/" + filename + "forecastarma.txt"
    expectedFileRMSE = "data/" + filename + "forecastarmaerrors.txt"
    data = read_file(expectedFile)
    if data != None:
        return HttpResponse(json.dumps(data), content_type="application/json")

    file = "data/"+filename+"workload.txt"
    data = read_json(file)
    data = collections.OrderedDict(sorted(data.items()))

    print data
    dates = data.keys()
    values = data.values()
    values = np.asfarray(values)
    initial_length = int(0.10 * len(values))

    forecast1 = UnivariateForecasting(values, len(values) - initial_length, initial_length)
    res = forecast1.compute_arma()
    result = {}
    rmse = {}
    rmse["ARMA"] = forecast1.compute_rmse()
    result["RMSE"] = rmse


    with open(expectedFileRMSE, 'w+') as outfile:
        json.dump(result, outfile)

    prediction = []
    for i in range(0, initial_length):
        prediction.append(values[i])

    prediction = np.append(prediction, res)

    result = {}
    for i in range(0, len(dates)):
        result[dates[i]] = prediction[i]
    result = collections.OrderedDict(sorted(result.items()))

    with open(expectedFile, 'w+') as outfile:
        json.dump(result, outfile)

    return HttpResponse(json.dumps(result), content_type="application/json")

def resources(request):
    filename = request.GET['log']
    expectedFile = "data/" + filename + "forecastarmaresources.txt"
    expectedFileRMSE = "data/" + filename + "forecastarmaerrorsresources.txt"
    data = read_file(expectedFile)
    if data != None:
        return HttpResponse(json.dumps(data), content_type="application/json")

    file = "data/"+filename+"resource.txt"
    data = read_json(file)
    data = collections.OrderedDict(sorted(data.items()))

    print data
    dates = data.keys()
    values = data.values()
    values = np.asfarray(values)
    initial_length = int(0.10 * len(values))

    forecast1 = UnivariateForecasting(values, len(values) - initial_length, initial_length)
    res = forecast1.compute_arma()
    result = {}
    rmse = {}
    rmse["ARMA"] = forecast1.compute_rmse()
    result["RMSE"] = rmse


    with open(expectedFileRMSE, 'w+') as outfile:
        json.dump(result, outfile)

    prediction = []
    for i in range(0, initial_length):
        prediction.append(values[i])

    prediction = np.append(prediction, res)

    result = {}
    for i in range(0, len(dates)):
        result[dates[i]] = prediction[i]
    result = collections.OrderedDict(sorted(result.items()))

    with open(expectedFile, 'w+') as outfile:
        json.dump(result, outfile)

    return HttpResponse(json.dumps(result), content_type="application/json")

def get_error(request):
    filename = request.GET['log']
    expectedFileRMSE = "data/" + filename + "forecastarmaerrors.txt"
    data = read_file(expectedFileRMSE)
    if data != None:
        return HttpResponse()

    return HttpResponse(json.dumps(data), content_type="application/json")

def read_json(path):
    with open(path) as data_file:
        data = json.load(data_file)
    return data

def read_file(filename):
    file_data = None
    if isfile(filename):
        file = open(filename)
        file_data = json.loads(file.read())
        file_data = collections.OrderedDict(sorted(file_data.items()))

    return file_data


