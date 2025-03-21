# Generated by Django 5.1 on 2025-02-19 05:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dbenter', '0005_cleaneddata_delete_csvrecord'),
    ]

    operations = [
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('username', models.CharField(max_length=100, unique=True)),
                ('email', models.EmailField(max_length=254, unique=True)),
                ('password', models.CharField(max_length=255)),
            ],
            options={
                'db_table': 'login_signup',
            },
        ),
    ]
