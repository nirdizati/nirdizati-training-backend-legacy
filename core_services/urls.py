from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^yolo', views.yolo, name='yolo'),
    url(r'^configer', views.run_configuration, name='configer'),
    url(r'^classConfiger', views.run_class_configuration, name='classConfiger'),
    url(r'^listAvailableResultsFiles', views.listAvailableResultsFiles, name='listAvailableResultsFiles'),
    url(r'^listAvailableResultsPrefix', views.listAvailableResultsPrefix, name='listAvailableResultsPrefix'),
    url(r'^listAvailableResultsLog', views.listAvailableResultsLog, name='listAvailableResultsLog'),
    url(r'^listAvailableRules', views.listAvailableRules, name='listAvailableRules'),
    url(r'^listAvailableThreshold', views.listAvailableThreshold, name='listAvailableThreshold'),
    url(r'^fileToJsonResults', views.fileToJsonResults, name='fileToJsonResults'),
    url(r'^fileToJsonGeneralResults', views.fileToJsonGeneralResults, name='fileToJsonGeneralResults'),
    url(r'^downloadCsv', views.downloadCsv, name='downloadCsv'),
    url(r'^getConfStatus', views.getConfStatus, name='getConfStatus'),
    url(r'^downloadZip', views.downloadZip, name='downloadZip')
    
    
    

]
