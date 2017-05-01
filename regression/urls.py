from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^index', views.index, name='index'),
    url(r'^linear', views.linear, name='linear'),
    url(r'^randomforestregression', views.randomforestregression, name='randomforestregression'),
    url(r'^xgboost', views.xgboost, name='xgboost'),

    url(r'^evaluationlinear', views.linearevaluation, name='linearevaluation'),
    url(r'^evaluationrandomforestregression', views.randomforestregressionevaluation, name='randomforestregressionevaluation'),
    url(r'^evaluationxgboost', views.xgboostevaluation, name='xgboostevaluation'),

    url(r'^generallinear', views.lineargeneral, name='lineargeneral'),
    url(r'^generalrandomforestregression', views.randomforestregressiongeneral, name='randomforestregressiongeneral'),
    url(r'^generalxgboost', views.xgboostgeneral, name='xgboostgeneral')
]