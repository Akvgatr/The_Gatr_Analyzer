from django.db import models
from tinymce.models import HTMLField
from autoslug import AutoSlugField
class News(models.Model):
    newstitle = models.CharField(max_length=100)
    newsdes = HTMLField()


    newsslug=AutoSlugField(populate_from='newstitle', unique=True,null=True,default=None)