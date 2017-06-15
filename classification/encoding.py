import csv
import datetime
import json
import time
from os.path import isfile

import pandas as pd
import untangle
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def remaining_time_encode(request):
    filename = request.GET['log']
    if isfile("encodedfiles/indexbased_"+filename):
        return HttpResponse()

    obj = untangle.parse('logdata/' + filename)

    traces = obj.log.trace

    header = ['id', 'remainingTime', 'elapsedTime', 'executedActivities']
    data = []

    events = get_all_events(traces)
    with open("encodedfiles/events_"+filename+".csv", "w") as output:
        writer = csv.writer(output, lineterminator='_')
        for val in events:
            writer.writerow([val])

    longest_trace = 0;
    for trace in traces:
        case_id = ''
        if type(trace.string) is list:
            for i in range(0, len(trace.string)):
                if u"concept:name" == trace.string[i]['key']:
                    case_id = trace.string[i]['value']
        else:
            # only has 1 value, so it automatically becomes the case id
            case_id = trace.string['value']

        #first event timestamp
        first_event = trace.event[0]
        first_event_timestamp = get_timestamp_from_event(first_event)

        #last event timestamp
        last_event = trace.event[len(trace.event) - 1]
        last_event_timestamp = get_timestamp_from_event(last_event)
        activity_name = ''
        activities_executed = 0
        for event in trace.event:
            for i in range(0, len(event.string)):
                if u"concept:name" == event.string[i]['key']:
                    activity_name += str(events.index(event.string[i]['value'])+1) + '_'
                    activities_executed += 1

            event_timestamp = get_timestamp_from_event(event)
            if event_timestamp == None:
                continue

            data.append([case_id, last_event_timestamp - event_timestamp, event_timestamp - first_event_timestamp, activities_executed, activity_name.rstrip('_')])
        if longest_trace < activities_executed:
            longest_trace = activities_executed
    #rewrite data
    new_data = []
    for d in data:
        history = d[len(d) - 1].split("_")
        history_data = []
        for i in range(0, longest_trace):
            if len(history) > i:
                history_data.append(history[i])
            else:
                history_data.append(0)

        new_data.append(d[0:4] + history_data)

    for i in range(0, longest_trace):
        header.append("prefix_"+str(i+1))

    df = pd.DataFrame(columns=header, data=new_data)
    write_pandas_to_csv(df, filename+'.csv')
    return HttpResponse(json.dumps(data), content_type="application/json")

def get_all_events(traces):
    events = []
    for trace in traces:
        for event in trace.event:
            for i in range(0, len(event.string)):
                if u"concept:name" == event.string[i]['key']:
                    activity_name = event.string[i]['value']
                    if activity_name not in events:
                        events.append(activity_name)
    return events

def get_timestamp_from_event(event):
    date_time = ''
    if event is None:
        return None
    if type(event.date) is list:
        for i in range(0, len(event.date)):
            if u"time:timestamp" == event.date[i]['key']:
                date_time = event.date[i]['value']
    else:
        date_time = event.date['value']

    timestamp = time.mktime(datetime.datetime.strptime(date_time[0:19], "%Y-%m-%dT%H:%M:%S").timetuple())

    return timestamp

def write_pandas_to_csv(df, filename):
    df.to_csv("encodedfiles/indexbased_"+filename,sep=',',mode='w+', index=False)
    return filename

def fast_slow_encode(request):
    filename = "encodedfiles/indexbased_"+request.GET['log']+".csv"
    if isfile(filename):
        prefix = int(request.GET['index']);
        df = pd.read_csv(filename)
        df = df[df['executedActivities'] == prefix+1]

        columns = len(df.columns)
        average_remaining_time = df["remainingTime"].mean()

        df['label'] = df["remainingTime"] < average_remaining_time
        df = df.drop('executedActivities', 1)
        df = df.drop('elapsedTime', 1)
        df = df.drop('remainingTime', 1)
        df = df.drop('id', 1)

        for i in range(1, columns):
            if i > prefix:
                try:
                    df = df.drop('prefix_'+str(i), 1)
                except:
                    print "column prefix_"+str(i)+" does not exist"
            else:
                df['prefix_'+str(i)].apply(str)

        df = df.sample(frac=1)

        return df
    return None

def ltl_encode(request):
    filename = "encodedfiles/indexbased_"+request.GET['log']+".csv"
    if isfile(filename):
        activityA = int(request.GET['activityA'])
        activityB = int(request.GET['activityB'])
        prefix = int(request.GET['index']);
        df = pd.read_csv(filename)
        unique_cases = df.id.unique()

        df['label'] = 0
        for case in unique_cases:
            case_df = df[df['id'] == case]
            maxLengthCase = case_df['executedActivities'].max()
            case_df = case_df[case_df['executedActivities'] == maxLengthCase]
            activityAHappened = False
            activityBHappenedAfter = False

            for i in range(1, maxLengthCase):
                if activityAHappened:
                    if activityB == case_df.iloc[0]['prefix_'+str(i)]:
                        activityBHappenedAfter = True
                if activityA == case_df.iloc[0]['prefix_'+str(i)]:
                    activityAHappened = True

            df.label[df['id'] == case] = activityAHappened & activityBHappenedAfter

        df = df[df['executedActivities'] == prefix+1]

        columns = len(df.columns)+10
        df = df.drop('executedActivities', 1)
        df = df.drop('elapsedTime', 1)
        df = df.drop('remainingTime', 1)
        df = df.drop('id', 1)

        for i in range(1, columns):
            if i > prefix:
                try:
                    df = df.drop('prefix_'+str(i), 1)
                except:
                    print "column prefix_"+str(i)+" does not exist"
            else:
                df['prefix_'+str(i)].apply(str)

        df = df.sample(frac=1)

        return df
    return None