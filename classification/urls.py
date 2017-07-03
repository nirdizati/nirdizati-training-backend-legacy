from django.conf.urls import url

from classification import views

urlpatterns = [
    url(r'^index', views.index, name='index'),
    url(r'^dt', views.dt, name='dt'),
    url(r'^rf', views.rf, name='rf'),
    url(r'^knn', views.rf, name='knn'),
]