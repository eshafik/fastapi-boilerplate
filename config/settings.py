# config/settings.py
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
dotenv_path = os.path.join(BASE_DIR, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

# Environment settings
FASTAPI_ENV = os.getenv('FASTAPI_ENV', 'development')
DEBUG = FASTAPI_ENV == 'development'

# Installed apps
INSTALLED_APPS = [
    'apps.app1',
    # Add more apps here
]

# Database settings
TORTOISE_ORM_CONFIG = {
    "connections": {
        "default": os.getenv("DATABASE_URL", "sqlite://db.sqlite3")
    },
    "apps": {
        "models": {
            "models": [],
            "default_connection": "default",
        }
    },
}

# Celery settings
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Logging settings (example)
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'DEBUG' if DEBUG else 'INFO',
    },
}
