from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^yolo', views.yolo, name='yolo'),
    url(r'^configer', views.run_configuration, name='configer'),
<<<<<<< HEAD
=======
    url(r'^classConfiger', views.run_class_configuration, name='classConfiger'),
>>>>>>> resolvehead
    url(r'^listAvailableResultsFiles', views.listAvailableResultsFiles,
        name='listAvailableResultsFiles'),
    url(r'^listAvailableResultsPrefix', views.listAvailableResultsPrefix,
        name='listAvailableResultsPrefix'),
    url(r'^listAvailableResultsLog', views.listAvailableResultsLog,
        name='listAvailableResultsLog'),
    url(r'^fileToJsonResults', views.fileToJsonResults,
        name='fileToJsonResults'),
<<<<<<< HEAD
=======
    url(r'^fileToJsonGeneralResults', views.fileToJsonGeneralResults,
        name='fileToJsonGeneralResults'),
>>>>>>> resolvehead
     url(r'^downloadCsv', views.downloadCsv,
        name='downloadCsv')
    

]
