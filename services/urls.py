from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^index', views.index, name='index'),
    url(r'^results$', views.results, name='results'),
    url(r'^general', views.get_general_evaluation, name='general'),
    url(r'^evaluation', views.get_evaluation, name='evaluation'),
]