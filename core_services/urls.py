from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^yolo', views.yolo, name='yolo'),
    url(r'^configer', views.run_configuration, name='configer'),
    url(r'^listAvailableResultsFiles', views.listAvailableResultsFiles,
        name='listAvailableResultsFiles'),
    url(r'^listAvailableResultsPrefix', views.listAvailableResultsPrefix,
        name='listAvailableResultsPrefix'),
    url(r'^listAvailableResultsLog', views.listAvailableResultsLog,
        name='listAvailableResultsLog'),
    url(r'^fileToJsonResults', views.fileToJsonResults,
        name='fileToJsonResults'),
     url(r'^downloadCsv', views.downloadCsv,
        name='downloadCsv')
    

]
