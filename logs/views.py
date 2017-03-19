import datetime
import json
import time

import pandas as pd
import untangle
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def index(request):
    results = {}
    results["message"] = "in logs"
    print 'creating data prep csv'

    prep_data('Production.xes')
    return HttpResponse(json.dumps(results), content_type="application/json")

def prep_data(filename):
    obj = untangle.parse(filename)

    traces = obj.log.trace

    header = ['id', 'startTime', 'event', 'resource', 'datetime', 'eventString', 'resourceString']
    data = []

    for trace in traces:
        case_id = trace.string[0]['value']
        for event in trace.event:
            activity_name = event.string[0]['value']
            date_time = event.date['value']
            start_time = time.mktime(datetime.datetime.strptime(date_time.split("+")[0], "%Y-%m-%dT%H:%M:%S.%f").timetuple())
            resource = event.string[4]['value']
            data.append([case_id, start_time, activity_name, resource, date_time, activity_name, resource])

    print 'done data preparation'

    df = pd.DataFrame(columns=header, data=data)
    unique_case_id = pd.Series(df.id).unique()
    df['id'].replace(unique_case_id, range(unique_case_id.size), inplace=True)

    unique_event = pd.Series(df.event).unique()
    df['event'].replace(unique_event, range(unique_event.size), inplace=True)

    unique_resource = pd.Series(df.resource).unique()
    df['resource'].replace(unique_resource, range(unique_resource.size), inplace=True)

    write_pandas_to_csv(df, '')

def write_pandas_to_csv(df, filename):
    filename = 'prepdata.csv'
    df.to_csv(filename,sep=',',mode='w')
    return filename

def handle_uploaded_file(f):
    print 'handle file upload'
    with open('sample.csv', 'wb+') as destination:
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
    workload = count_active_resources('Production.xes')
    return HttpResponse(json.dumps(workload), content_type="application/json")

def count_active_resources(filename):
    obj = untangle.parse(filename)

    traces = obj.log.trace
    resources_per_day = {}
    workload = {}
    dates = []

    for trace in traces:
        for event in trace.event:
            resources = []
            resource = event.string[4]['value']
            date_time = event.date['value'].split("T")[0]
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
        print key
        print value
        workload[key] = len(value)

    return workload

@csrf_exempt
def traces(request):
    workload = count_active_traces('Production.xes')

    return HttpResponse(json.dumps(workload), content_type="application/json")

def count_active_traces(filename):
    obj = untangle.parse(filename)

    traces = obj.log.trace
    workload = {}
    dates = []

    for trace in traces:
        active_dates = []
        for event in trace.event:
            date_time = event.date['value'].split("T")[0]
            if not date_time in active_dates:
                active_dates.append(date_time)
            else:
                continue

            if not date_time in dates:
                dates.append(date_time)
                workload[date_time] = 1
            else:
                workload[date_time] = workload[date_time] + 1

    return workload

@csrf_exempt
def event_executions(request):
    workload = count_event_executions('Production.xes')

    return HttpResponse(json.dumps(workload), content_type="application/json")

def count_event_executions(filename):
    obj = untangle.parse(filename)

    traces = obj.log.trace
    executions = {}
    events = []

    for trace in traces:
        for event in trace.event:
            activity_name = event.string[0]['value']

            if not activity_name in events:
                events.append(activity_name)
                executions[activity_name] = 1
            else:
                executions[activity_name] = executions[activity_name] + 1

    return executions
