from django_rq import job
from core_services import regression


@job("high", timeout='1h') # timeout is optional
def regressionTask(fileName, prefix, encoding, cluster, regressionType):
	if regressionType == 'linear':
		regression.linear(fileName, prefix, encoding, cluster)
	if regressionType == 'randomforest':
		regression.randomforestregression(fileName, prefix, encoding, cluster)
	if regressionType == 'xgboost':
		regression.xgboost(fileName, prefix, encoding, cluster)
	