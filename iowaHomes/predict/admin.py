from django.contrib import admin

from predict.models import UserRole
from predict.models import User
from predict.models import HouseListings

admin.site.register(UserRole)
admin.site.register(User)
admin.site.register(HouseListings)