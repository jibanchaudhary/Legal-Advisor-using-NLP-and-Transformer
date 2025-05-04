from django.urls import path
from . import views

app_name = 'advisory'

urlpatterns = [
    path('', views.index, name='index'),  # This is the index page for the advisory app
    path('advisory/',views.advisory_query,name ="advisory_query"),
    path('query_result/',views.query_result,name="query_result"),
    path('irrelevant_query/',views.irrelevant_query,name ="irrelevant_query"),
]
