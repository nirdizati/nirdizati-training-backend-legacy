import datetime
import json
import time

import pandas as pd
import untangle
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from os import listdir
from os.path import isfile, join

import collections


@csrf_exempt
def index(request):
    results = {}
    results["message"] = "in logs"
    print 'creating data prep csv'

    # prep_data('logdata/bpi2013_20.xes')
    prep_data('logdata/' + request.GET['log'])
    return HttpResponse(json.dumps(results), content_type="application/json")

def prep_data(filename):
    obj = untangle.parse(filename)

    traces = obj.log.trace

    # header = ['id', 'startTime', 'event', 'resource', 'datetime', 'eventString', 'resourceString']
    header = ['id', 'startTime', 'event', 'resource']
    data = []

    for trace in traces:
        case_id = 'traceId'
        if type(trace.string) is list :
            for i in range(0, len(trace.string)):
                if u"concept:name" == trace.string[i]['key']:
                    case_id = trace.string[i]['value']
        else:
            #only has 1 value, so it automatically becomes the case id
            case_id = trace.string['value']

        for event in trace.event:
            activity_name = 'eventName'
            resource = 'resource'
            for i in range(0, len(event.string)):
                if u"concept:name" == event.string[i]['key']:
                    activity_name = event.string[i]['value']
                if u"Resource" == event.string[i]['key']:
                    resource = event.string[i]['value']
                elif u"org:resource" == event.string[i]['key']:
                    resource = event.string[i]['value']
            date_time = ''
            if type(event.date) is list:
                for i in range(0, len(event.date)):
                    if u"time:timestamp" == event.date[i]['key']:
                        date_time = event.date[i]['value']
            else:
                date_time = event.date['value']
            start_time = time.mktime(datetime.datetime.strptime(date_time[0:19], "%Y-%m-%dT%H:%M:%S").timetuple())
            data.append([case_id, start_time, activity_name, resource])
            # data.append([case_id, start_time, activity_name, resource, date_time, activity_name, resource])

    print 'done data preparation'

    df = pd.DataFrame(columns=header, data=data)
    unique_case_id = pd.Series(df.id).unique()
    df['id'].replace(unique_case_id, range(unique_case_id.size), inplace=True)

    unique_event = pd.Series(df.event).unique()
    df['event'].replace(unique_event, range(unique_event.size), inplace=True)

    unique_resource = pd.Series(df.resource).unique()
    df['resource'].replace(unique_resource, range(unique_resource.size), inplace=True)

    write_pandas_to_csv(df, filename)

def write_pandas_to_csv(df, filename):
    filename = filename+'.csv'
    df.to_csv(filename,sep=',',mode='w+', index=False)
    return filename

def handle_uploaded_file(f):
    print 'handle file upload'
    with open('logdata/'+f.name, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

@csrf_exempt
def process_log(request):
    response_data = {}
    response_data['result'] = 'error'
    response_data['message'] = 'No request'

    print request.method
    if request.method == 'POST':
        handle_uploaded_file(request.FILES['file'])
        filename = request.FILES['file'].name

        prep_data(filename)
        return HttpResponse(json.dumps(response_data), content_type="application/json")
    else:
        return HttpResponse(json.dumps(response_data), content_type="application/json")

def read_from_query(path):
    df = pd.read_csv(filepath_or_buffer=path, header=0, index_col=0)  # ,nrows = 1000)
    return df

@csrf_exempt
def resources(request):
    workload = count_active_resources(request.GET['log'])
    return HttpResponse(json.dumps(workload), content_type="application/json")

def count_active_resources(filename):
    expectedFile = "data/" + filename + "resource.txt"
    data = read_file(expectedFile)
    if data != None:
        return data
    obj = untangle.parse('logdata/' + filename)

    traces = obj.log.trace
    resources_per_day = {}
    workload = {}
    dates = []

    for trace in traces:
        for event in trace.event:
            resources = []
            resource = 'resource'
            for i in range(0, len(event.string)):
                if u"Resource" == event.string[i]['key']:
                    resource = event.string[i]['value']
                elif u"org:resource" == event.string[i]['key']:
                    resource = event.string[i]['value']
            date_time = ''
            if type(event.date) is list:
                for i in range(0, len(event.date)):
                    if u"time:timestamp" == event.date[i]['key']:
                        date_time = event.date[i]['value']
            else:
                date_time = event.date['value']
            date_time = date_time.split("T")[0]
            resources.append(resource)
            if date_time in dates:
                current_resources = resources_per_day[date_time]
                if not resource in current_resources:
                    current_resources.append(resource)
                resources_per_day[date_time] = current_resources
            else:
                dates.append(date_time)
                resources_per_day[date_time] = resources

    for key, value in resources_per_day.iteritems():
        workload[key] = len(value)

    workload = collections.OrderedDict(sorted(workload.items()))
    with open(expectedFile, 'w+') as outfile:
        json.dump(workload, outfile)
    return workload

@csrf_exempt
def traces(request):
    workload = count_active_traces(request.GET['log'])
    return HttpResponse(json.dumps(workload), content_type="application/json")

def count_active_traces(filename):
    expectedFile = "data/" + filename + "workload.txt"
    data = read_file(expectedFile)
    if data != None:
        print data
        return data
    obj = untangle.parse('logdata/' + filename)

    traces = obj.log.trace
    workload = {}
    dates = []

    for trace in traces:
        active_dates = []
        for event in trace.event:
            date_time = ''
            if type(event.date) is list:
                for i in range(0, len(event.date)):
                    if u"time:timestamp" == event.date[i]['key']:
                        date_time = event.date[i]['value']
            else:
                date_time = event.date['value']
            date_time = date_time.split("T")[0]
            if not date_time in active_dates:
                active_dates.append(date_time)
            else:
                continue

            if not date_time in dates:
                dates.append(date_time)
                workload[date_time] = 1
            else:
                workload[date_time] = workload[date_time] + 1

    workload = collections.OrderedDict(sorted(workload.items()))
    with open(expectedFile, 'w+') as outfile:
        json.dump(workload, outfile)
    return workload

@csrf_exempt
def event_executions(request):
    filename = request.GET['log'];
    workload = count_event_executions(filename)

    return HttpResponse(json.dumps(workload), content_type="application/json")

def count_event_executions(filename):
    expectedFile = "data/" + filename + "eventexecutions.txt"
    data = read_file(expectedFile)
    if data != None:
        return data
    obj = untangle.parse('logdata/'+filename)

    traces = obj.log.trace
    executions = {}
    events = []

    for trace in traces:
        for event in trace.event:
            activity_name = 'eventName'
            for i in range(0, len(event.string)):
                if u"concept:name" == event.string[i]['key']:
                    activity_name = event.string[i]['value']
            if not activity_name in events:
                events.append(activity_name)
                executions[activity_name] = 1
            else:
                executions[activity_name] = executions[activity_name] + 1

    executions = collections.OrderedDict(sorted(executions.items(), reverse=True, key=executions.get))
    with open(expectedFile, 'w+') as outfile:
        json.dump(executions, outfile)
    return executions

def read_file(filename):
    file_data = None
    if isfile(filename):
        file = open(filename)
        file_data = json.loads(file.read())
        file_data = collections.OrderedDict(sorted(file_data.items()))

    return file_data

@csrf_exempt
def list_log_files(request):
    path = 'logdata'
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    onlyfiles = filter(lambda x: x.endswith((".xes")), onlyfiles)
    return HttpResponse(json.dumps(onlyfiles), content_type="application/json")

@csrf_exempt
def upload_file(request):
    message = {}
    if request.method == 'POST':
        handle_uploaded_file(request.FILES['file'])
        filename = request.FILES['file'].name
        message['success'] = "true"
        message['filename'] = filename
        return HttpResponse(json.dumps(message), content_type="application/json")
    else:
        message['success'] = "false"
        return HttpResponse(json.dumps(message), content_type="application/json")