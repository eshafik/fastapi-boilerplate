# config/celery.py
from celery import Celery, signals

from config.db import init_db
from config.settings import CELERY_BROKER_URL, CELERY_RESULT_BACKEND, INSTALLED_APPS

celery_app = Celery('fastapi_tortoise_boilerplate', broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)


# Auto-discover tasks from all apps
celery_app.autodiscover_tasks([app for app in INSTALLED_APPS])
celery_app.conf.update(
    result_expires=3600,
)


# Initialize Tortoise ORM before any task starts
@signals.worker_process_init.connect
def init_worker(**kwargs):
    import asyncio
    asyncio.run(init_db())
