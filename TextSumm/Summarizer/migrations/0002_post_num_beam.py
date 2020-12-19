# Generated by Django 3.1.3 on 2020-11-28 03:23

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Summarizer', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='post',
            name='num_beam',
            field=models.IntegerField(default=1, validators=[django.core.validators.MaxValueValidator(10), django.core.validators.MinValueValidator(1)]),
        ),
    ]