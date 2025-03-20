from django.contrib import admin

# Register your models here.
from services.models import services
class serviceadmin(admin.ModelAdmin):
    list_display=('serviceicon','servicetitle','servicedes')

admin.site.register(services,serviceadmin)