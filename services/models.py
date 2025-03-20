from django.db import models

# Create your models here.
class services(models.Model):#shouldbe Model
    serviceicon=models.CharField(max_length=50)
    servicetitle=models.CharField(max_length=50)
    servicedes=models.CharField(max_length=50)