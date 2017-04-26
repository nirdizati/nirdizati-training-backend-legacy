from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^index', views.index, name='index'),
    url(r'^workload', views.workload, name='workload'),
    url(r'^resources', views.resources, name='resources'),
    url(r'^errors', views.get_error, name='errors')
]