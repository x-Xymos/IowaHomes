# Generated by Django 2.2.4 on 2019-08-10 18:46

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='UserRole',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=64)),
            ],
        ),
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=64)),
                ('email', models.CharField(max_length=64)),
                ('phone', models.IntegerField(default=0)),
                ('role', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='predict.UserRole')),
            ],
        ),
        migrations.CreateModel(
            name='HouseListings',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('MSZoning', models.CharField(max_length=2)),
                ('Neighborhood', models.CharField(max_length=16)),
                ('Bedroom', models.IntegerField(default=0)),
                ('LotArea', models.IntegerField(default=0)),
                ('SalePrice', models.IntegerField(default=0)),
                ('Kitchen', models.IntegerField(default=0)),
                ('FirstFlrSF', models.IntegerField(default=0)),
                ('SecondFlrSF', models.IntegerField(default=0)),
                ('pub_date', models.DateTimeField(verbose_name='date published')),
                ('agent', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='predict.User')),
            ],
        ),
    ]
