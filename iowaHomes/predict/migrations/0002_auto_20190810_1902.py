# Generated by Django 2.2.4 on 2019-08-10 19:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predict', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='houselistings',
            name='address',
            field=models.CharField(default='', max_length=128),
        ),
        migrations.AlterField(
            model_name='user',
            name='phone',
            field=models.CharField(max_length=15),
        ),
    ]
