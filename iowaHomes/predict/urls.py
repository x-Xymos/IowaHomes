from django.urls import path

from . import views


app_name = 'predict'
urlpatterns = [
    path('', views.IndexView.as_view(), name='index'),
    path('estimate/', views.estimate, name='estimate'),
    path('browse/', views.BrowseView.as_view(), name='browse'),
    path('login/', views.LoginView.as_view(), name='login'),
]