import json
import os
from os.path import isfile

import csv

import django_rq
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponseBadRequest


<<<<<<< HEAD
from core_services import encoding, regression
=======
from core_services import encoding, prediction
>>>>>>> resolvehead
from project import tasks
import pandas as pd

import numpy as np
from sklearn.cluster import KMeans
<<<<<<< HEAD
# regression.linear("Production.xes", 5, 'boolean', "sd")
=======
from sklearn.linear_model import Lasso
from sklearn.cluster import DBSCAN

>>>>>>> resolvehead


def index(request):
    return HttpResponse()


def yolo(request):
<<<<<<< HEAD
    df = pd.read_csv(filepath_or_buffer='core_encodedFiles/boolean_Production.xes_10.csv', header=0)
    
    estimator = KMeans(n_clusters=3)    
    estimator.fit(df)
    lists = {i: df.iloc[np.where(estimator.labels_ == i)[0]] for i in range(estimator.n_clusters)}
    print lists[0]
    # encoding.encode("Production.xes", 9)

    # #regression.linear("Production.xes", 9, 'xg', "sd")
=======
    encoding.encode("Production.xes", 5)
    
    prediction.regressior("Production.xes", 5, 'simpleIndex', "Kmeans", 'linear')
    # prediction.classifier("Production.xes", 5, 'simpleIndex', "Kmeans", 'KNN')
    # prediction.classifier("Production.xes", 5, 'simpleIndex', "Kmeans", 'RandomForest')
    # prediction.classifier("Production.xes", 5, 'simpleIndex', "Kmeans", 'DecisionTree')

    # prediction.classifier("Production.xes", 13, 'simpleIndex', "None", 'KNN')
    # prediction.classifier("Production.xes", 13, 'simpleIndex', "None", 'RandomForest')
    # prediction.classifier("Production.xes", 13, 'simpleIndex', "None", 'DecisionTree')



    #regression.linear()

    # df = pd.read_csv(filepath_or_buffer='core_encodedFiles/simpleIndex_Production.xes_16.csv', header=0)
    # data_ = df[["Id", "remainingTime"]]

    # estimator = DBSCAN(eps=0.3, min_samples=10,  metric='haversine')    
    # estimator.fit(data_)
    # print estimator
    # labels = estimator.labels_
    # n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # print n_clusters
    # cluster_lists = {i: df.iloc[np.where(estimator.labels_ == i)[0]] for i in range(n_clusters)}
    # print len(cluster_lists)    
    # writeHeader = True
    # for cluster_list in cluster_lists:
    #     clusterd_data = cluster_lists[cluster_list]
    #     original_cluster_data = cluster_lists[cluster_list]
    #     lm = Lasso(fit_intercept=True, warm_start=True)
    #     y = clusterd_data['remainingTime']
    #     clusterd_data = clusterd_data.drop('remainingTime', 1)
    #     lm.fit(clusterd_data, y)
    #     original_cluster_data['prediction'] = lm.predict(clusterd_data)
    #     if writeHeader is True:
    #         original_cluster_data.to_csv('core_results/cluster.csv', sep=',', mode='a', header=True, index=False)
    #         writeHeader = False
    #     else:
    #         original_cluster_data.to_csv('core_results/cluster.csv', sep=',', mode='a', header=False, index=False)
            

    # 

    #regression.linear("Production.xes", 9, 'xg', "sd")
>>>>>>> resolvehead
    # # regression.linear("Production.xes", 5, 'simple_index', "sd")
    # regression.xgboost("Production.xes", 9, 'simpleIndex', "sd")
    # # fileName, prefix, encoding, cluster, regression
    # # django_rq.enqueue(tasks.regressionTask,"Production.xes", 5, 'simple_index', "sd", "xgboost")
    return HttpResponse("YOLO")


def listAvailableResultsFiles(request):
    log = request.GET['log']
    prefix = request.GET['Prefix']
<<<<<<< HEAD
    path = "core_results/" + log + "/" + prefix
=======
    res_type = request.GET['restype']
    # res_type = "_class"
    path = "core_results" + res_type + '/' + log + "/" + prefix
>>>>>>> resolvehead
    try:
        files = os.listdir(path)
        return HttpResponse(json.dumps(files), content_type="application/json")
    except OSError as exc:  # Guard against race condition
        return HttpResponse("No Results")

def listAvailableResultsPrefix(request):
    log = request.GET['log']
<<<<<<< HEAD
    path = "core_results/" + log
=======
    res_type = request.GET['restype']

    path = "core_results" + res_type + '/' + log
>>>>>>> resolvehead
    try:
        files = os.listdir(path)
        return HttpResponse(json.dumps(files), content_type="application/json")
    except OSError as exc:  # Guard against race condition
        return HttpResponse("No Results")

def listAvailableResultsLog(request):
<<<<<<< HEAD
    path = "core_results/"
=======
    res_type = request.GET['restype']
    path = "core_results" + res_type

>>>>>>> resolvehead
    try:
        files = os.listdir(path)
        return HttpResponse(json.dumps(files), content_type="application/json")
    except OSError as exc:  # Guard against race condition
        return HttpResponse("No Results")

<<<<<<< HEAD

=======
def fileToJsonGeneralResults(request):
    log = request.GET['log']
    prefix = request.GET['Prefix']
    res_type = request.GET['restype']

    expected_filename = 'core_results'+ res_type + '/' + log + '/' + str(prefix) + '/General.csv'
    data = resultsAsJson(expected_filename)
    if data is not None:
        return HttpResponse(data, content_type="application/json")
    else:
        return HttpResponseBadRequest()
    
>>>>>>> resolvehead
def fileToJsonResults(request):
    log = request.GET['log']
    prefix = request.GET['Prefix']
    encoding = request.GET['encoding']
    regMethod = request.GET['method']
<<<<<<< HEAD
    expected_filename = 'core_results/' + log + '/' + str(prefix) + '/' + regMethod + '_' + encoding + '.csv'
=======
    cluster = request.GET['cluster']
    res_type = request.GET['restype']

    expected_filename = 'core_results'+ res_type + '/' + log + '/' + str(prefix) + '/' + regMethod + '_' + encoding + '_'  + cluster + '_clustering.csv'
>>>>>>> resolvehead
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
<<<<<<< HEAD
            for regression in configuration_json['regression']:
                django_rq.enqueue(tasks.regressionTask, log,
                                  prefix, encodingMethod, "sd", regression)
=======
            for clustering in configuration_json['clustering']:
                for regression in configuration_json['regression']:
                    django_rq.enqueue(tasks.regressionTask, log,
                                      prefix, encodingMethod, clustering, regression)
    return HttpResponse("YOLO")

@csrf_exempt
def run_class_configuration(request):
    if request.method == 'POST':
        configuration_json = json.loads(request.body)
        print configuration_json
        log = configuration_json["log"]
        prefix = configuration_json['prefix']
        # Encode the file.
        encoding.encode(log, prefix)
        for encodingMethod in configuration_json['encoding']:
            for clustering in configuration_json['clustering']:
                for classification in configuration_json['classification']:
                    django_rq.enqueue(tasks.classifierTask, log,
                                      prefix, encodingMethod, clustering, classification)
>>>>>>> resolvehead
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