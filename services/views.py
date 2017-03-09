import json
from django.http import HttpResponse

def index(request):
	response_data = {}
	response_data['result'] = 'error'
	response_data['message'] = 'Some error message'

	return HttpResponse(json.dumps(response_data), content_type="application/json")
    # return HttpResponse("Hello, world. You're at the polls index.")