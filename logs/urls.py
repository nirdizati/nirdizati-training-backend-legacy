from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^index', views.index, name='index'),
    url(r'^process', views.process_log, name='process'),
    url(r'^traces', views.traces, name='traces'),
    url(r'^resources', views.resources, name='resources'),
    url(r'^events', views.event_executions, name='events'),
    url(r'^list', views.list_log_files, name='list'),
    url(r'^upload', views.upload_file, name='upload'),
]