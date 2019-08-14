from django.urls import path

from . import views


app_name = 'predict'
urlpatterns = [
    path('', views.IndexView.as_view(), name='index'),
    path('estimate/', views.EstimateView, name='estimate'),
    path('browse/', views.BrowseView.as_view(), name='browse'),
]