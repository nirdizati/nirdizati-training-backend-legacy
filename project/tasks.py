from django_rq import job
from core_services import prediction
import pandas as pd
import numpy as np
import time
from pydblite import Base





@job("high", timeout='1h') # timeout is optional
def regressionTask(fileName, prefix, encoding, cluster, regressionType):
	db = Base('backendDB.pdl')
	db.open()
	run = regressionType + '_' + encoding + '_' + cluster

	records =  [r for r in db if r['Run'] == run and r['Prefix'] == str(prefix) and r['Log'] == fileName]
	db.update(records[0], TimeStamp=time.strftime("%b %d %Y %H:%M:%S", time.localtime()), Status='Running')
	db.commit()
	try:
		prediction.regressior(fileName, prefix, encoding, cluster, regressionType)
		records = [r for r in db if r['Run'] == run and r['Prefix'] == str(prefix) and r['Log'] == fileName]
		db.update(records[0], TimeStamp=time.strftime("%b %d %Y %H:%M:%S", time.localtime()), Status='Completed')
		db.commit()

	except Exception,e:  # Guard against race condition
		records = [r for r in db if r['Run'] == run and r['Prefix'] == str(prefix) and r['Log'] == fileName]
		db.update(records[0], TimeStamp=time.strftime("%b %d %Y %H:%M:%S", time.localtime()), Status='Failed: ' + str(e))
		db.commit()

		raise

@job("high", timeout='1h') # timeout is optional
def classifierTask(fileName, prefix, encoding, cluster, method, label, threshold):
	db = Base('backendDB.pdl')
	db.open()
	run = method + '_' + encoding + '_' + cluster + '_' + label +  '_' + str(threshold)
	#time.sleep(2)
	records = [r for r in db if r['Run'] == run and r['Prefix'] == str(prefix) and r['Log'] == fileName and r['Rule'] == label and r['Threshold'] == str(threshold)]
	db.update(records[0], TimeStamp=time.strftime("%b %d %Y %H:%M:%S", time.localtime()), Status='Running')
	db.commit()
	try:
		prediction.classifier(fileName, prefix, encoding, cluster, method, label, threshold)
		records = [r for r in db if r['Run'] == run and r['Prefix'] == str(prefix) and r['Log'] == fileName]
		db.update(records[0], TimeStamp=time.strftime("%b %d %Y %H:%M:%S", time.localtime()), Status='Completed')
		db.commit()

	except Exception,e:  # Guard against race condition
		records = [r for r in db if r['Run'] == run and r['Prefix'] == str(prefix) and r['Log'] == fileName and r['Rule'] == label and r['Threshold'] == str(threshold)]
		db.update(records[0], TimeStamp=time.strftime("%b %d %Y %H:%M:%S", time.localtime()), Status='Failed: ' + str(e))
		db.commit()
		raise
