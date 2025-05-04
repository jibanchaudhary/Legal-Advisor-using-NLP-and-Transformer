from django.apps import AppConfig

class AdvisoryConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'advisory'

    def ready(self):
        print("Advisory app is ready, but GPT-2 model is not preloaded.")
