# config/settings.py
import os
from typing import Optional

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
    'apps.user',
    'apps.document',
    'apps.conversation',
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
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/1')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY')
OPENAI_EMBEDDING_MODEL: str = os.getenv('OPENAI_EMBEDDING_MODEL', "text-embedding-3-small")
OPENAI_CHAT_MODEL: str = os.getenv('OPENAI_CHAT_MODEL', "GPT-3.5")

# Elasticsearch
ELASTICSEARCH_URL: str = os.getenv('ELASTICSEARCH_URL', "http://localhost:9200")
ELASTICSEARCH_API_KEY: str = os.getenv('ELASTICSEARCH_API_KEY', None)
ELASTICSEARCH_USERNAME: Optional[str] = os.getenv('ELASTICSEARCH_USERNAME', None)
ELASTICSEARCH_PASSWORD: Optional[str] = os.getenv('ELASTICSEARCH_PASSWORD', None)
ELASTICSEARCH_DEFAULT_INDEX_NAME = os.getenv('ELASTICSEARCH_DEFAULT_INDEX_NAME', "test_index")

CHUNK_SIZE: int = int(os.getenv('CHUNK_SIZE', '1000'))
CHUNK_OVERLAP: int = int(os.getenv('CHUNK_OVERLAP', '200'))

# Ingestion limits
MAX_PDF_PAGES: int = int(os.getenv('MAX_PDF_PAGES', '100'))
MAX_WEB_PAGES: int = int(os.getenv('MAX_WEB_PAGES', '50'))


MAX_CONVERSATION_HISTORY: int = int(os.getenv('MAX_CONVERSATION_HISTORY', '10'))
DEFAULT_MAX_CHUNKS: int = int(os.getenv('DEFAULT_MAX_CHUNKS', '5'))
DEFAULT_TEMPERATURE: float = float(os.getenv('DEFAULT_TEMPERATURE', '0.7'))
DEFAULT_MAX_TOKENS: int = int(os.getenv('DEFAULT_MAX_TOKENS', '1000'))

# Memory Configuration
CHUNK_MEMORY_DECAY_DAYS: int = int(os.getenv('CHUNK_MEMORY_DECAY_DAYS', '7'))
MAX_CHUNK_REUSE_COUNT: int = int(os.getenv('MAX_CHUNK_REUSE_COUNT', '3'))

# Performance Configuration
CHAT_TIMEOUT_SECONDS: int = int(os.getenv('CHAT_TIMEOUT_SECONDS', '30'))
MAX_CONCURRENT_CHATS: int = int(os.getenv('MAX_CONCURRENT_CHATS', '100'))

# Rate Limiting
RATE_LIMIT_PER_USER_PER_MINUTE: int = int(os.getenv('RATE_LIMIT_PER_USER_PER_MINUTE', '20'))
RATE_LIMIT_PER_USER_PER_HOUR: int = int(os.getenv('RATE_LIMIT_PER_USER_PER_HOUR', '100'))


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
