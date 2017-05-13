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
        df = df[df['executedActivities'] == prefix]
        columns = len(df.columns)
        for i in range(1, columns):
            if i > prefix + 1:
                try:
                    df = df.drop('prefix_'+str(i), 1)
                except:
                    print "column prefix_"+str(i)+" does not exist"

        return HttpResponse(df.to_csv(index = False))
    return HttpResponse("File not found")