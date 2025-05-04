from django.db import models

# feedback/models.py

from django.db import models


class Feedback(models.Model):
    name = models.CharField(max_length=255)
    email = models.EmailField()
    phone = models.CharField(max_length=15)
    company = models.CharField(max_length=255)
    country = models.CharField(max_length=255)
    job_title = models.CharField(max_length=255)
    queries = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Feedback from {self.name}"
