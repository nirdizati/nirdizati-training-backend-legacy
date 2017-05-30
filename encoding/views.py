import json

from django.http import HttpResponse
from os.path import isfile
import pandas as pd

def index(request):
    return HttpResponse()

def read(request):
    filename = "encodedfiles/indexbased_"+request.GET['log']+".csv"
    if isfile(filename):
        prefix = int(request.GET['index']);
        df = pd.read_csv(filename)
        df = df[df['executedActivities'] == prefix+1]

        columns = len(df.columns)

        df = df.drop('executedActivities', 1)
        df = df.drop('elapsedTime', 1)
        df = df.drop('remainingTime', 1)
        df = df.drop('id', 1)

        for i in range(1, columns):
            if i > prefix + 1:
                try:
                    df = df.drop('prefix_'+str(i), 1)
                except:
                    print "column prefix_"+str(i)+" does not exist"
            else:
                df['prefix_'+str(i)].apply(str)

        return HttpResponse(df.to_csv(index = False))
    return HttpResponse("File not found")

def events(request):
    filename = "encodedfiles/events_"+request.GET['log']+".csv"

    if isfile(filename):
        file = open(filename, "r")
        events = file.read().split("_")
        return HttpResponse(json.dumps(events))
    return HttpResponse("file not found")


def fast_slow_encode(request):
    import encoding
    return encoding.fast_slow_encode(request)

def ltl_encode(request):
    import encoding
    return encoding.ltl_encode(request)