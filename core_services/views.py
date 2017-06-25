import json
import os

import django_rq
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from core_services import encoding, regression
from project import tasks


# regression.linear("Production.xes", 5, 'boolean', "sd")


def index(request):
    return HttpResponse()


def yolo(request):
    encoding.encode("Production.xes", 5)

    regression.linear("Production.xes", 5, 'frequency', "sd")
    # regression.linear("Production.xes", 5, 'simple_index', "sd")
    # regression.xgboost("Production.xes", 5, 'simple_index', "sd")
    # fileName, prefix, encoding, cluster, regression
    # django_rq.enqueue(tasks.regressionTask,"Production.xes", 5, 'simple_index', "sd", "xgboost")
    return HttpResponse("YOLO")


def listAvailableResultsFiles(request):
    log = request.GET['log']
    prefix = request.GET['Prefix']
    path = "core_results/" + log + "/" + prefix
    try:
        files = os.listdir(path)
        return HttpResponse(files)
    except OSError as exc:  # Guard against race condition
        return HttpResponse("No Results")

def listAvailableResultsPrefix(request):
    log = request.GET['log']
    path = "core_results/" + log
    try:
        files = os.listdir(path)
        return HttpResponse(files)
    except OSError as exc:  # Guard against race condition
        return HttpResponse("No Results")

def listAvailableResultsLog(request):
    path = "core_results/"
    try:
        files = os.listdir(path)
        return HttpResponse(files)
    except OSError as exc:  # Guard against race condition
        return HttpResponse("No Results")



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
