from django_rq import job
<<<<<<< HEAD
from core_services import regression
=======
from core_services import prediction
>>>>>>> resolvehead


@job("high", timeout='1h') # timeout is optional
def regressionTask(fileName, prefix, encoding, cluster, regressionType):
<<<<<<< HEAD
	if regressionType == 'linear':
		regression.linear(fileName, prefix, encoding, cluster)
	if regressionType == 'randomforest':
		regression.randomforestregression(fileName, prefix, encoding, cluster)
	if regressionType == 'xgboost':
		regression.xgboost(fileName, prefix, encoding, cluster)
=======
	prediction.regressior(fileName, prefix, encoding, cluster, regressionType)


@job("high", timeout='1h') # timeout is optional
def classifierTask(fileName, prefix, encoding, cluster, method):
	prediction.classifier(fileName, prefix, encoding, cluster, method)
>>>>>>> resolvehead
	