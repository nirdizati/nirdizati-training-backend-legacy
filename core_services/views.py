import json
import os
from os.path import isfile

import csv

import django_rq
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponseBadRequest


from core_services import encoding, regression
from project import tasks
import pandas as pd

import numpy as np
from sklearn.cluster import KMeans
# regression.linear("Production.xes", 5, 'boolean', "sd")


def index(request):
    return HttpResponse()


def yolo(request):
    df = pd.read_csv(filepath_or_buffer='core_encodedFiles/boolean_Production.xes_10.csv', header=0)
    
    estimator = KMeans(n_clusters=3)    
    estimator.fit(df)
    lists = {i: df.iloc[np.where(estimator.labels_ == i)[0]] for i in range(estimator.n_clusters)}
    print lists[0]
    # encoding.encode("Production.xes", 9)

    # #regression.linear("Production.xes", 9, 'xg', "sd")
    # # regression.linear("Production.xes", 5, 'simple_index', "sd")
    # regression.xgboost("Production.xes", 9, 'simpleIndex', "sd")
    # # fileName, prefix, encoding, cluster, regression
    # # django_rq.enqueue(tasks.regressionTask,"Production.xes", 5, 'simple_index', "sd", "xgboost")
    return HttpResponse("YOLO")


def listAvailableResultsFiles(request):
    log = request.GET['log']
    prefix = request.GET['Prefix']
    path = "core_results/" + log + "/" + prefix
    try:
        files = os.listdir(path)
        return HttpResponse(json.dumps(files), content_type="application/json")
    except OSError as exc:  # Guard against race condition
        return HttpResponse("No Results")

def listAvailableResultsPrefix(request):
    log = request.GET['log']
    path = "core_results/" + log
    try:
        files = os.listdir(path)
        return HttpResponse(json.dumps(files), content_type="application/json")
    except OSError as exc:  # Guard against race condition
        return HttpResponse("No Results")

def listAvailableResultsLog(request):
    path = "core_results/"
    try:
        files = os.listdir(path)
        return HttpResponse(json.dumps(files), content_type="application/json")
    except OSError as exc:  # Guard against race condition
        return HttpResponse("No Results")


def fileToJsonResults(request):
    log = request.GET['log']
    prefix = request.GET['Prefix']
    encoding = request.GET['encoding']
    regMethod = request.GET['method']
    expected_filename = 'core_results/' + log + '/' + str(prefix) + '/' + regMethod + '_' + encoding + '.csv'
    data = resultsAsJson(expected_filename)
    if data is not None:
        return HttpResponse(data, content_type="application/json")
    else:
        return HttpResponseBadRequest()


@csrf_exempt
def run_configuration(request):
    if request.method == 'POST':
        configuration_json = json.loads(request.body)
        print configuration_json
        log = configuration_json["log"]
        prefix = configuration_json['prefix']
        # Encode the file.
        encoding.encode(log, prefix)
        for encodingMethod in configuration_json['encoding']:
            for regression in configuration_json['regression']:
                django_rq.enqueue(tasks.regressionTask, log,
                                  prefix, encodingMethod, "sd", regression)
    return HttpResponse("YOLO")


def resultsAsJson(expected_filename):
    if isfile(expected_filename):
        with open(expected_filename) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        jsonData = json.dumps(rows)
        return jsonData
    return None

def downloadCsv(request):
    log = request.GET['log']
    prefix = request.GET['Prefix']
    encoding = request.GET['encoding']
    regMethod = request.GET['method']
    expected_filename = 'core_results/' + log + '/' + str(prefix) + '/' + regMethod + '_' + encoding + '.csv'
    with open(expected_filename, 'rb') as myfile:
        response = HttpResponse(myfile, content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename='+ expected_filename
    
    return response