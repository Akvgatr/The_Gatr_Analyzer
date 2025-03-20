"""
URL configuration for project1 project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from project1 import views
from dbenter.views import upload_csv ,display_csv , select_scraped_table, scrape_tables, display_scraped_table# Import the upload view
from .views import download_csv
from .views import view_user_logs
from .views import predictive_analysis
from .views import advanced_preprocessing

from . import views  

from django.urls import path
from dbenter.views import get_unique_values



urlpatterns = [

    path("signup/", views.signup, name="signup"),
    path("login/", views.login_view, name="login"),
    path("", views.homepage, name="homepage"),

    path("logout/", views.logout_view, name="logout"),
    path("download_csv/", download_csv, name="download_csv"),


    
    path('admin/', admin.site.urls),
   
    path('upload/', upload_csv, name='upload_csv'),  # Add this line
    path('display_csv/', display_csv, name='display_csv'),  # Add this line
   


    path('analysis/', views.analysis, name='analysis'),
    path("sql_analysis/", views.analyze_sql_query, name="sql_analysis"), 


    path('save_in_database/', views.save_in_database, name='save_in_database'),



    path('graphs/', views.graphs_view, name='graphs'),

    path('generate_graph/', views.generate_graph, name='generate_graph'),  # Ensure this exists


    path("view_logs/", view_user_logs, name="view_logs"),
    path("dashboard/", views.dashboard_view, name="dashboard"),


    path("scrape-tables/", scrape_tables, name="scrape_tables"),
    path("select-table/", select_scraped_table, name="select_scraped_table"),
    path("display-scraped-table/", display_scraped_table, name="display_scraped_table"),


    path("advanced_preprocessing/", advanced_preprocessing, name="advanced_preprocessing"),

    path("predictive_analysis/", predictive_analysis, name="predictive_analysis"),
    path("get_unique_values/", get_unique_values, name="get_unique_values"),


]


    
