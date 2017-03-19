from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^process', views.process_log, name='process'),
    url(r'^traces', views.traces, name='traces'),
    url(r'^resources', views.resources, name='resources'),
    url(r'^events', views.event_executions, name='events'),
]