from django.conf.urls import url

from encoding import remaining_time_encode
from . import views

urlpatterns = [
    url(r'^index', views.index, name='index'),
    url(r'^encode', remaining_time_encode, name='encode'),
    url(r'^read', views.read, name='read'),
    url(r'^events', views.events, name='events'),
    url(r'^fastslowencode', views.fast_slow_encode, name='fastslowencode'),
    url(r'^ltlencode', views.ltl_encode, name='ltlencode'),
]