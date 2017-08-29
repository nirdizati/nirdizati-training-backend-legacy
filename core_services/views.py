import json
import os
from os.path import isfile

import csv

import django_rq
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponseBadRequest

import zipfile
import StringIO
from core_services import encoding, prediction
from project import tasks
import pandas as pd
import time
from pydblite import Base


import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso
from sklearn.cluster import DBSCAN



def index(request):
    return HttpResponse()

def getConfStatus(request):
    db = Base('backendDB.pdl')
    if db.exists(): 
        db.open()
    records = [];
    for r in db:
        print r
        records.append(r)

    return HttpResponse(json.dumps(records), content_type="application/json")


def yolo(request):
    db = Base('backendDB.pdl')
    db.create('Type','Log', 'Run', 'Prefix','Rule','Threshold', 'TimeStamp', 'Status', mode="override")

    # if db.exists(): 
    #     db.open()
    records = [];
    for r in db:
        records.append(r)
    db.delete(records)
    db.commit()
    # encoding.encode("SepsisCasesEventLog.xes", 8)
    
    # prediction.regressior("SepsisCasesEventLog.xes", 8, 'complexIndex', "Kmeans", 'linear')
    # prediction.classifier("SepsisCasesEventLog.xes", 8, 'simpleIndex', "None", 'DecisionTree', 'duration', 'default')
    # prediction.classifier("SepsisCasesEventLog.xes", 8, 'simpleIndex', "Kmeans", 'DecisionTree', 'duration', 'default')
    # prediction.classifier("SepsisCasesEventLog.xes", 8, 'boolean', "Kmeans", 'DecisionTree', 'duration', 'default')

    # prediction.classifier("SepsisCasesEventLog.xes", 8, 'simpleIndex', "None", 'DecisionTree', 'remainingTime', 'default')

    # prediction.classifier("Production.xes", 5, 'simpleIndex', "Kmeans", 'RandomForest')
    # prediction.classifier("Production.xes", 5, 'simpleIndex', "Kmeans", 'DecisionTree')

    # prediction.classifier("Production.xes", 13, 'simpleIndex', "None", 'KNN')
    # prediction.classifier("Production.xes", 13, 'simpleIndex', "None", 'RandomForest')
    # prediction.classifier("Production.xes", 3, 'complexIndex', "Kmeans", 'DecisionTree')



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
    # # regression.linear("Production.xes", 5, 'simple_index', "sd")
    # regression.xgboost("Production.xes", 9, 'simpleIndex', "sd")
    # # fileName, prefix, encoding, cluster, regression
    # # django_rq.enqueue(tasks.regressionTask,"Production.xes", 5, 'simple_index', "sd", "xgboost")
    return HttpResponse("YOLO")


def listAvailableResultsFiles(request):
    log = request.GET['log']
    prefix = request.GET['Prefix']
    res_type = request.GET['restype']
    if res_type == '_class':
        rule = request.GET['rule']
        threshold = request.GET['threshold']
        path = 'core_results'+ res_type + '/' + log + '/' + str(prefix) + '/' + rule + '/' + str(threshold)

    else:
        path = "core_results" + res_type + '/' + log + "/" + prefix
    try:
        files = os.listdir(path)
        return HttpResponse(json.dumps(files), content_type="application/json")
    except OSError as exc:  # Guard against race condition
        return HttpResponse("No Results")

def listAvailableResultsPrefix(request):
    log = request.GET['log']
    res_type = request.GET['restype']

    path = "core_results" + res_type + '/' + log
    try:
        files = os.listdir(path)
        return HttpResponse(json.dumps(files), content_type="application/json")
    except OSError as exc:  # Guard against race condition
        return HttpResponse("No Results")


def listAvailableRules(request):
    log = request.GET['log']
    res_type = request.GET['restype']
    prefix = request.GET['Prefix']

    path = "core_results" + res_type + '/' + log + '/' + prefix
    try:
        files = os.listdir(path)
        return HttpResponse(json.dumps(files), content_type="application/json")
    except OSError as exc:  # Guard against race condition
        return HttpResponse("No Results")

def listAvailableThreshold(request):
    log = request.GET['log']
    res_type = request.GET['restype']
    prefix = request.GET['Prefix']
    rule = request.GET['rule']

    path = "core_results" + res_type + '/' + log + '/' + prefix + '/' + rule
    try:
        files = os.listdir(path)
        return HttpResponse(json.dumps(files), content_type="application/json")
    except OSError as exc:  # Guard against race condition
        return HttpResponse("No Results")


def listAvailableResultsLog(request):
    res_type = request.GET['restype']
    path = "core_results" + res_type

    try:
        files = os.listdir(path)
        return HttpResponse(json.dumps(files), content_type="application/json")
    except OSError as exc:  # Guard against race condition
        return HttpResponse("No Results")

def fileToJsonGeneralResults(request):
    log = request.GET['log']
    prefix = request.GET['Prefix']
    res_type = request.GET['restype']

    if res_type == '_class':
        rule = request.GET['rule']
        threshold = request.GET['threshold']
        expected_filename = 'core_results'+ res_type + '/' + log + '/' + str(prefix) + '/' + rule + '/' + str(threshold) + '/General.csv'

    else:
        expected_filename = 'core_results'+ res_type + '/' + log + '/' + str(prefix) + '/General.csv'
    
    data = resultsAsJson(expected_filename)
    if data is not None:
        return HttpResponse(data, content_type="application/json")
    else:
        return HttpResponseBadRequest()
    
def fileToJsonResults(request):
    log = request.GET['log']
    prefix = request.GET['Prefix']
    encoding = request.GET['encoding']
    regMethod = request.GET['method']
    cluster = request.GET['cluster']
    res_type = request.GET['restype']

    if res_type == '_class':
        rule = request.GET['rule']
        threshold = request.GET['threshold']
        expected_filename = 'core_results'+ res_type + '/' + log + '/' + str(prefix) + '/' + rule + '/' + str(threshold)  + '/' + regMethod + '_' + encoding + '_'  + cluster + '_clustering.csv'

    else:
        expected_filename = 'core_results'+ res_type + '/' + log + '/' + str(prefix) + '/' + regMethod + '_' + encoding + '_'  + cluster + '_clustering.csv'
    
    data = resultsAsJson(expected_filename)
    if data is not None:
        return HttpResponse(data, content_type="application/json")
    else:
        return HttpResponseBadRequest()



@csrf_exempt
def run_configuration(request):
    if request.method == 'POST':
        db = Base('backendDB.pdl')
        # db.create('Type','Log', 'Run', 'Prefix','Rule','Threshold', 'TimeStamp', 'Status', mode="override")
        if db.exists():
            db.open()
        else: 
            db.create('Type','Log', 'Run', 'Prefix','Rule','Threshold', 'TimeStamp', 'Status')


        configuration_json = json.loads(request.body)
        print configuration_json
        log = configuration_json["log"]
        prefix = configuration_json['prefix']
        # Encode the file.
        encoding.encode(log, prefix)
        
        for encodingMethod in configuration_json['encoding']:
            for clustering in configuration_json['clustering']:
                for regression in configuration_json['regression']:
                    django_rq.enqueue(tasks.regressionTask, log,
                                      prefix, encodingMethod, clustering, regression)
                    run = regression + '_' + encodingMethod + '_' + clustering
                    records = [r for r in db if r['Run'] == run and r['Prefix'] == str(prefix) and r['Log'] == log]
                    # for r in db:
                    #     if (r['Run'] == run) and (r['Prefix'] == str(prefix)) and (r['Log'] == log):
                    #         records.append(r)
                    print records
                    if not records:
                        db.insert("Regression", log, run, str(prefix),"NaN","NaN", time.strftime("%b %d %Y %H:%M:%S", time.localtime()), 'queued')
                    else:
                        db.update(records[0], TimeStamp=time.strftime("%b %d %Y %H:%M:%S", time.localtime()), Status= 'queued')
                    # if run in df['Run'].unique():
                    #     df.loc[df.Run == run, 'TimeStamp'] = time.strftime("%b %d %Y %H:%M:%S", time.localtime())
                    #     df.loc[df.Run == run, 'Status'] = "queued"
                    # else: 
                    #     df.loc[df.shape[0]] = [run, time.strftime("%b %d %Y %H:%M:%S", time.localtime()), 'queued']
        # print df
        # print df['Run'] 
        # df.to_csv('core_results_queue/' + log + '/' + str(prefix) + '/reg_queueStatus.csv', sep=',',header=writeHeader, mode='w+', index=False)
        db.commit()
    return HttpResponse("YOLO")

@csrf_exempt
def run_class_configuration(request):
    if request.method == 'POST':
        db = Base('backendDB.pdl')
        if db.exists():
            db.open()
        else: 
            db.create('Type','Log', 'Run', 'Prefix','Rule','Threshold', 'TimeStamp', 'Status')

        configuration_json = json.loads(request.body)
        print configuration_json
        log = configuration_json["log"]
        prefix = configuration_json['prefix']
        rule = configuration_json['rule']
        threshold = configuration_json['threshold']

        # Encode the file.
        encoding.encode(log, prefix)
        for encodingMethod in configuration_json['encoding']:
            for clustering in configuration_json['clustering']:
                for classification in configuration_json['classification']:
                    django_rq.enqueue(tasks.classifierTask, log,
                                      prefix, encodingMethod, clustering, classification, rule, threshold)
                    run = classification + '_' + encodingMethod + '_' + clustering
                    records = [r for r in db if r['Run'] == run and r['Prefix'] == str(prefix) and r['Log'] == log and r['Rule'] == rule and r['Threshold'] == str(threshold)]
                    print records
                    if not records:
                        db.insert("Classification", log, run, str(prefix), rule, str(threshold), time.strftime("%b %d %Y %H:%M:%S", time.localtime()), 'queued')
                    else:
                        db.update(records[0], TimeStamp=time.strftime("%b %d %Y %H:%M:%S", time.localtime()), Status= 'queued')
        db.commit()        
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

def make_dir(drpath):
    if not os.path.exists(drpath):
        try:
            os.makedirs(drpath)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def downloadZip(request):
    log = request.GET['log']
    prefix = request.GET['Prefix']
    res_type = request.GET['restype']

    if res_type == 'class':
        rule = request.GET['rule']
        threshold = request.GET['threshold']
        expected_filename = 'core_results_'+ res_type + '/' + log + '/' + str(prefix) + '/' + rule + '/' + str(threshold)
        zip_subdir = res_type + '_' + log + '_' + str(prefix) + '_' + rule + '_' + str(threshold) 
    else:
        expected_filename = 'core_results_'+ res_type + '/' + log + '/' + str(prefix)
        zip_subdir = res_type + '_' + log + '_' + str(prefix)

    files = os.listdir(expected_filename)
    # Files (local path) to put in the .zip
    # FIXME: Change this (get paths from DB etc)
    zip_filename = "%s.zip" % zip_subdir
    # Folder name in ZIP archive which contains the above files
    # E.g [thearchive.zip]/somefiles/file2.txt
    # FIXME: Set this to something better

    # Open StringIO to grab in-memory ZIP contents
    s = StringIO.StringIO()

    # The zip compressor
    zf = zipfile.ZipFile(s, "w")

    for fpath in files:
        zip_path = os.path.join(expected_filename, fpath)

        # Add file, at correct path
        zf.write(expected_filename +'/' + fpath, zip_path)

    # Must close zip for all contents to be written
    zf.close()

    # Grab ZIP file from in-memory, make response with correct MIME-type
    resp = HttpResponse(s.getvalue(), content_type= "application/x-zip-compressed")
    # ..and correct content-disposition
    resp['Content-Disposition'] = 'attachment; filename=%s' % zip_filename

    return resp