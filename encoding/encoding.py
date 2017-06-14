import csv
import datetime
import json
import time
import os
from os.path import isfile
import pynq as pynq
import pandas as pd
import numpy as np
import untangle
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

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

#C:\\thesis\\predict-python\\encodedfiles\\bool_encoded_traces.csv
obj = untangle.parse('C:\\thesis\\predict-python\\logdata\\Production.xes')
prefix_ = 5;
attrs_dictionary = {};
unique_events_resource = list()

bool_encoded_traces = list();
freq_encoded_traces = list();
complex_index_traces = list();
latest_payload_index_traces = list();
simple_index_traces = list()

event_attr_names = list()
event_attr_names_header = list()
event_attr_latest_names = list()
events_header = list()
key_value = 0;

for event_attr in obj.log.trace[0].event[0].string[0:12]:
    event_attr_latest_names.append("Latest_" + event_attr['key'])
    event_attr_names.append(event_attr['key'])


for i in xrange(1,prefix_ + 1):
    events_header.append("Event_" +  str(i))
    for attr in event_attr_names:
        event_attr_names_header.append(attr + "_" + str(i))

for trace in obj.log.trace:
    if len(trace.event) > prefix_:
        for event in trace.event[0:prefix_]:
            if(event.string[2]['value'] not in unique_events_resource):
                unique_events_resource.append(event.string[2]['value']) 



for trace in obj.log.trace:
    if len(trace.event) >= prefix_:
        bool_trace = list();
        freq_trace = list();
        trace_event_ids = list();

        bool_trace.append(trace.string[0]['value'])
        freq_trace.append(trace.string[0]['value'])
        trace_events = list();
        last_event_timestamp_ = get_timestamp_from_event(trace.event[len(trace.event)-1])
        last_prefix_remainingTime = get_timestamp_from_event(trace.event[prefix_ - 1])
        last_prefix_event_attr = list()
        event_attr_values = list();

        for event_attr in trace.event[prefix_ - 1].string[0:12]:
                if event_attr['value'] not in attrs_dictionary:
                    attrs_dictionary[event_attr['value']] = key_value
                    key_value = key_value + 1;
                last_prefix_event_attr.append(attrs_dictionary[event_attr['value']])
       
        for event in trace.event[0:prefix_]:
            if event.string[2]['value'] not in attrs_dictionary:
                    attrs_dictionary[event.string[2]['value']] = key_value
                    key_value = key_value + 1;
            trace_event_ids.append(attrs_dictionary[event.string[2]['value']])
            event_timestamp = get_timestamp_from_event(event)
            for event_attr in event.string[0:12]:
                if event_attr['value'] not in attrs_dictionary:
                    attrs_dictionary[event_attr['value']] = key_value
                    key_value = key_value + 1;
                event_attr_values.append(attrs_dictionary[event_attr['value']])

            trace_events.append(event.string[2]['value'])

        complex_index_traces.append([trace.string[0]['value']] + event_attr_values + [last_event_timestamp_ - last_prefix_remainingTime])
        latest_payload_index_traces.append([trace.string[0]['value']] + trace_event_ids + last_prefix_event_attr + [last_event_timestamp_ - last_prefix_remainingTime])
        simple_index_traces.append([trace.string[0]['value']] + trace_event_ids + [last_event_timestamp_ - last_prefix_remainingTime])
        
        #Bool and Freq encoding
        for unique_id in unique_events_resource:
            if (unique_id in trace_events):
                bool_trace.append("1")
                freq_trace.append(trace_events.count(unique_id))
            else: 
                bool_trace.append("0")
                freq_trace.append("0")
        bool_encoded_traces.append(bool_trace + [last_event_timestamp_ - last_prefix_remainingTime])
        freq_encoded_traces.append(freq_trace + [last_event_timestamp_ - last_prefix_remainingTime])            

with open("C:\\thesis\\predict-python\\encodedfiles\\bool_encoded_traces.csv", 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(["caseId"] + unique_events_resource + ["remainingTime"])
    for x in bool_encoded_traces:
        wr.writerow(x)

with open("C:\\thesis\\predict-python\\encodedfiles\\freq_encoded_traces.csv", 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(["caseId"] + unique_events_resource)
    for x in freq_encoded_traces:
        wr.writerow(x)

with open("C:\\thesis\\predict-python\\encodedfiles\\complex_index.csv", 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(["caseId"] + event_attr_names_header + ["remainingTime"])
    for x in complex_index_traces:
        wr.writerow(x)

with open("C:\\thesis\\predict-python\\encodedfiles\\simple_index.csv", 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(["caseId"] + events_header + ["remainingTime"])
    for x in simple_index_traces:
        wr.writerow(x)

with open("C:\\thesis\\predict-python\\encodedfiles\\index_latest_payload.csv", 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(["caseId"] + events_header + event_attr_latest_names + ["remainingTime"])
    for x in latest_payload_index_traces:
        wr.writerow(x)

print attrs_dictionary

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

def bool_freq_encode(request):
    filename = request.GET['log']
    prefix_ = request.GET['prefix']
    encodingmethod = request.GET['encodingmethod']
    # if isfile("encodedfiles/indexbased_"+filename):
    #     return HttpResponse()

    obj = untangle.parse('logdata/' + filename)

    unique_events_resource = list()
    for trace in obj.log.trace:
        if len(trace.event) > prefix_:
            for event in trace.event[0:prefix_]:
                if(event.string[2]['value'] not in unique_events_resource):
                    unique_events_resource.append(event.string[2]['value']) 

    bool_encoded_traces = list();
    freq_encoded_traces = list();

    for trace in obj.log.trace:
        if len(trace.event) >= prefix_:
            bool_trace = list();
            freq_trace = list();
            bool_trace.append(trace.string[0]['value'])
            freq_trace.append(trace.string[0]['value'])
            trace_events = list();
            last_event_timestamp_ = get_timestamp_from_event(trace.event[len(trace.event)-1])
            last_prefix_remainingTime = get_timestamp_from_event(trace.event[prefix_ - 1])

            for event in trace.event[0:prefix_]:
                   trace_events.append(event.string[2]['value'])
            for unique_id in unique_events_resource:
                if (unique_id in trace_events):
                    bool_trace.append("1")
                    freq_trace.append(trace_events.count(unique_id))
                else: 
                    bool_trace.append("0")
                    freq_trace.append("0")
            bool_encoded_traces.append(bool_trace + [last_event_timestamp_ - last_prefix_remainingTime])
            freq_encoded_traces.append(freq_trace + [last_event_timestamp_ - last_prefix_remainingTime])            

    with open("encodedfiles/" + filename + "bool_encoded_traces.csv", 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["caseId"] + unique_events_resource + ["remainingTime"])
        for x in bool_encoded_traces:
            wr.writerow(x)

    with open("encodedfiles/" + filename + "freq_encoded_traces.csv", 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["caseId"] + unique_events_resource)
        for x in freq_encoded_traces:
            wr.writerow(x)
    
    if (encodingmethod == "boolean"):
        return HttpResponse(json.dumps(bool_encoded_traces), content_type="application/json")
    else:
        return HttpResponse(json.dumps(freq_encoded_traces), content_type="application/json")

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

        return HttpResponse(df.to_csv(index = False))
    return HttpResponse("File not found")

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

        return HttpResponse(df.to_csv(index = False))
    return HttpResponse("File not found")
