# Generated by Django 5.1 on 2024-08-29 12:42

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='services',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('serviceicon', models.CharField(max_length=50)),
                ('servicetitle', models.CharField(max_length=50)),
                ('servicedes', models.CharField(max_length=50)),
            ],
        ),
    ]
 