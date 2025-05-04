from django.urls import path
from email_integration import views
from .views import *


app_name = 'email_integration'
urlpatterns = [
    path('', views.index, name='index'),
    path('record/',views.record_page, name = 'record_page'),
    path('record-audio/', views.record_audio, name='record_audio'),
    path('send-email/', views.send_transcribed_email, name='send_transcribed_email'),
]
