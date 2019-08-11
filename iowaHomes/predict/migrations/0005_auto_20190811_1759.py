# Generated by Django 2.2.4 on 2019-08-11 17:59

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('predict', '0004_auto_20190811_1257'),
    ]

    operations = [
        migrations.AddField(
            model_name='houselistings',
            name='image',
            field=models.ImageField(blank=True, null=True, upload_to='static/uploads'),
        ),
        migrations.AlterField(
            model_name='houselistings',
            name='agent',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL),
        ),
        migrations.DeleteModel(
            name='User',
        ),
        migrations.DeleteModel(
            name='UserRole',
        ),
    ]
