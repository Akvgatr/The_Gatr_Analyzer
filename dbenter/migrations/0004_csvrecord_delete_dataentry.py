# Generated by Django 5.1 on 2024-10-02 19:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dbenter', '0003_dataentry'),
    ]

    operations = [
        migrations.CreateModel(
            name='CSVRecord',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('data', models.JSONField()),
            ],
        ),
        migrations.DeleteModel(
            name='DataEntry',
        ),
    ]
