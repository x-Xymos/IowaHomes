from django.db import models
from django.contrib.auth.models import User



class HouseListings(models.Model):

    agent = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    address = models.CharField(max_length=128, default="")
    Bedroom = models.IntegerField(default=0)
    Kitchen = models.IntegerField(default=0)
    LotArea = models.IntegerField(default=0)
    SalePrice = models.IntegerField(default=0)
    FirstFlrSF = models.IntegerField(default=0)
    SecondFlrSF = models.IntegerField(default=0)
    pub_date = models.DateTimeField('date published')
    image = models.ImageField(upload_to='static/uploads', blank=True, null=True)

    def __str__(self):
        return 'HouseListing: ' + str(self.id)






