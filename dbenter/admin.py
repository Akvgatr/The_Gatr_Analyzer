from django.contrib import admin
import pandas as pd

# Register your models here.
from dbenter.models import dbenter
class dbenteradmin(admin.ModelAdmin):
    list_display=('name','author')

admin.site.register(dbenter,dbenteradmin)


from .models import CleanedData

admin.site.register(CleanedData)
