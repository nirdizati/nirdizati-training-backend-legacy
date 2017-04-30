import datetime
import json
import time

import pandas as pd
import untangle
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def remaining_time_encode(request):
    filename = request.GET['log']
    obj = untangle.parse('logdata/' + filename)

    traces = obj.log.trace

    header = ['id', 'remainingTime', 'elapsedTime', 'history']
    data = []

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
        for event in trace.event:
            for i in range(0, len(event.string)):
                if u"concept:name" == event.string[i]['key']:
                    activity_name += event.string[i]['value'] + '_'

            event_timestamp = get_timestamp_from_event(event)

            data.append([case_id, last_event_timestamp - event_timestamp, event_timestamp - first_event_timestamp, activity_name.rstrip('_')])

    # data = {'a':'b'}
    df = pd.DataFrame(columns=header, data=data)
    write_pandas_to_csv(df, filename+'.csv')
    return HttpResponse(json.dumps(data), content_type="application/json")

def get_timestamp_from_event(event):
    date_time = ''
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