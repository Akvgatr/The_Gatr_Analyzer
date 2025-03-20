from django.db import models
from django.contrib.auth.models import User  
from django.db import models
from django.db.models import JSONField  

class dbenter(models.Model):
    name = models.CharField(max_length=50)
    author = models.CharField(max_length=50)





# class DataAnalysis(models.Model):
#     user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)  # Temporarily allow null
#     table_name = models.CharField(max_length=255, null=True, blank=True)  # Allow null for now
#     data = models.JSONField()

#     def __str__(self):
#         return f"{self.user.username if self.user else 'No User'} - {self.table_name if self.table_name else 'No Table'}"

class DataAnalysis(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)  
    table_name = models.CharField(max_length=255, null=True, blank=True)
    data = models.JSONField()

    def __str__(self):
        return f"{self.user.username if self.user else 'No User'} - {self.table_name if self.table_name else 'No Table'}"




class CleanedData(models.Model):
    cleaned_data = models.JSONField()  

    def __str__(self):
        return str(self.cleaned_data)  


class User(models.Model):
    username = models.CharField(max_length=100, unique=True)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=255)  

    class Meta:
        db_table = "login_signup"  

    def __str__(self):
        return self.username







