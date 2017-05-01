from django.conf.urls import url

from . import views
from . import encoding

urlpatterns = [
    url(r'^index', views.index, name='index'),
    url(r'^workload', views.workload, name='workload'),
    url(r'^resources', views.resources, name='resources'),
    url(r'^errors', views.get_error, name='errors'),
    url(r'^forecast', views.forecast_remaining_time, name='forecast'),
    url(r'^encode', encoding.remaining_time_encode, name='encode')
]