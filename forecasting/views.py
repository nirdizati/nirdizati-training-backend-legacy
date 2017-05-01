import collections
import json
from os.path import isfile

import numpy as np
from django.http import HttpResponse
from django.http import HttpResponseBadRequest

from univariate_forecasting import UnivariateForecasting
import pandas as pd

from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

def forecast_remaining_time(request):
    filename = request.GET['log']

    expectedFile = "Results/forecasting_"+filename+'.csv'
    if isfile(expectedFile):
        data = pd.read_csv(expectedFile)
        data['remainingTime'] = data['remainingTime']/3600
        data['prediction'] = data['prediction'] / 3600

        data = data[data.remainingTime != data.prediction]

        return HttpResponse(data.to_json(), content_type="application/json")

    df = pd.read_csv(filepath_or_buffer="encodedfiles/indexbased_"+filename+'.csv', header=0)

    df['prediction'] = df['remainingTime']
    traces = df['id'].unique()
    starting_index = 5

    for trace in traces:
        trace_remaining_time = df.remainingTime[df['id'] == trace]
        index_values = trace_remaining_time.index.tolist()
        trace_remaining_time_reset = trace_remaining_time.reset_index(drop=True)
        if len(trace_remaining_time_reset) < starting_index:
            continue

        forecast1 = UnivariateForecasting(trace_remaining_time_reset, len(trace_remaining_time_reset) - starting_index, starting_index)
        res = forecast1.compute_arma()

        i = 0
        for result in res:
            df['prediction'][index_values[i+starting_index]] = result
            i += 1

    df = df.drop('history', 1)
    df = df.drop('elapsedTime', 1)
    write_pandas_to_csv(df, filename+'.csv')

    df = df[df.remainingTime != df.prediction]
    return HttpResponse(df.to_json(), content_type="application/json")

def forecast_remaining_time_general(request):
    filename = request.GET['log']

    expectedFile = "Results/forecasting_"+filename+'.csv'
    data = pd.read_csv(expectedFile)
    if not data.empty:
        data['remainingTime'] = data['remainingTime']/3600
        data['prediction'] = data['prediction'] / 3600

        data = data[data.remainingTime != data.prediction]

        results = {}
        results['RMSE'] = root_mean_square_error_calculation(data, 'remainingTime', 'prediction')
        results['MAE'] = mean_absolute_error_calculation(data, 'remainingTime', 'prediction')

        return HttpResponse(json.dumps(results), content_type="application/json")

    return HttpResponseBadRequest()

def forecast_remaining_time_evaluation(request):
    filename = request.GET['log']

    expectedFile = "Results/forecasting_"+filename+'.csv'
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
        return HttpResponse(json.dumps(results), content_type="application/json")

    return HttpResponseBadRequest()

def write_pandas_to_csv(df, filename):
    df.to_csv("Results/forecasting_"+filename,sep=',',mode='w+', index=False)
    return filename

def root_mean_square_error_calculation(df, field1, field2):
    return sqrt(mean_squared_error(df[field1], df[field2]))

def mean_absolute_error_calculation(df, field1, field2):
    return mean_absolute_error(df[field1], df[field2])