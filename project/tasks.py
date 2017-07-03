from django_rq import job
from core_services import prediction


@job("high", timeout='1h') # timeout is optional
def regressionTask(fileName, prefix, encoding, cluster, regressionType):
	prediction.regressior(fileName, prefix, encoding, cluster, regressionType)


@job("high", timeout='1h') # timeout is optional
def classifierTask(fileName, prefix, encoding, cluster, method):
	prediction.classifier(fileName, prefix, encoding, cluster, method)
	