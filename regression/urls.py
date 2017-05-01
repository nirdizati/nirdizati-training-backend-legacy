from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^index', views.index, name='index'),
    url(r'^linear', views.linear, name='linear'),
    url(r'^randomforestregression', views.randomforestregression, name='randomforestregression'),
    url(r'^xgboost', views.xgboost, name='xgboost'),
    url(r'^general', views.general, name='general'),
    url(r'^evaluation', views.evaluation, name='evaluation')
]