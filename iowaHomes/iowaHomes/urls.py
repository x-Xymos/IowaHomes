from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('predict/', include('predict.urls')),
    path('admin/', admin.site.urls),
]