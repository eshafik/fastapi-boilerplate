Welcome to FastAPI Boilerplate, a organized structure for rapid API development inspired by Django's efficiency. 
Utilizing Tortoise ORM, it mirrors the seamless ORM experience of Django while enabling easy integration 
of Celery for background task management. With intuitive CLI commands like 'python manage.py runserver',
'python manage.py startapp' and so on. Jumpstart your projects with ease.

Key Features:

- Django-like ORM experience with Tortoise ORM
- Celery integration for asynchronous task handling
- Simplified app management with CLI commands
- Fast and efficient API development
- Get started quickly—pull the repository and elevate your development workflow today! 

# Project Structure:
```bash
project/
├── apps/
│   ├── user/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── routers.py
│   │   ├── views.py
│   │   └── tasks.py
│   ├── app2/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── routers.py
│   │   ├── views.py
│   │   └── tasks.py
│   └── ...
├── config/
│   ├── __init__.py
│   ├── celery.py
│   ├── settings.py
│   └── tortoise.py
├── main.py
└── manage.py
```

## Create New App
```python manage.py startapp app_name```

## Register New App
```config/settings.py:```
```python
# Installed apps
INSTALLED_APPS = [
    'apps.user',
    # Add more apps here
]
```


## Run app
```python manage.py runserver ```

### Additional parameters:
```python manage.py runserver --host 0.0.0.0 --port 8000```

## Run script
```python manage.py runscript pyscript.py```

## Run Celery
```python manage.py runcelery```