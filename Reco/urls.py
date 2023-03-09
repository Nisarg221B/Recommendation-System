from django.conf.urls import url
from django.urls.resolvers import URLPattern
from . import views
from django.urls import path
from django.contrib import admin
app_name = 'Reco'
urlpatterns = [
    path('restaurant', views.showRest, name='showRest'),
    path('Menu', views.showMenu, name='showMenu'),
    path('profile', views.profileView, name='profileView'),
    path('Order', views.orderView, name='orderView'),
    path('Rate', views.rateView, name='rateView'),
]
