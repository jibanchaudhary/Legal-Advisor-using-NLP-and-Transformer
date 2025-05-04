from django.urls import path
from email_integration import views
from .views import *

app_name = 'users'
urlpatterns = [
    path("submit_feedback/", submit_feedback, name="submit_feedback"),
    path('login/', user_login, name='user_login'),
    path('logout/', user_logout, name='user_logout'),
    path('register/', user_register, name='user_register'),
]