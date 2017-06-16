from django_rq import job     

@job("high", timeout=600) # timeout is optional
def fooo():
     print "yoloy" # do some logic