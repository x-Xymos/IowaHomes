from django.db import models

class UserRole(models.Model):
    name = models.CharField(max_length=64)

    def __str__(self):
        return 'UserRole: ' + str(self.name)

class User(models.Model):
    role = models.ForeignKey(UserRole, on_delete=models.SET_NULL, null=True)
    name = models.CharField(max_length=64)
    email = models.CharField(max_length=64)
    phone = models.CharField(max_length=15)

    def __str__(self):
        return 'User: ' + str(self.name)


class HouseListings(models.Model):
    agent = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    address = models.CharField(max_length=128, default="")
    MSZoning = models.CharField(max_length=2)
    Neighborhood = models.CharField(max_length=16)
    Bedroom = models.IntegerField(default=0)
    LotArea = models.IntegerField(default=0)
    SalePrice = models.IntegerField(default=0)
    Kitchen = models.IntegerField(default=0)
    FirstFlrSF = models.IntegerField(default=0)
    SecondFlrSF = models.IntegerField(default=0)
    pub_date = models.DateTimeField('date published')

    def __str__(self):
        return 'HouseListing: ' + str(self.id)







