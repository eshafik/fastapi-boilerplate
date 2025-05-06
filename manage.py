#!/usr/bin/env python
import os
import click
import uvicorn
from config.celery import celery_app
import sys
from fastapi import FastAPI
from tortoise import Tortoise, run_async

from config.db import init_db
from config.settings import TORTOISE_ORM_CONFIG, INSTALLED_APPS, DEBUG

APP_TEMPLATE = """
# {app_name}/__init__.py
# This file makes {app_name} a Python package
"""

MODELS_TEMPLATE = """
# {app_name}/models.py
from tortoise import fields, models


class ExampleModel(models.Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=255)

    class Meta:
        default_connection = "default"
        table = "{app_name}_test_example_table"
"""

ROUTERS_TEMPLATE = """
# {app_name}/routers.py
from fastapi import APIRouter
from utils.response_wrapper import response_wrapper
from .views import example_view

router = APIRouter()

router.get("/{app_name}/example")(response_wrapper(example_view))
"""

VIEWS_TEMPLATE = """
# {app_name}/views.py
from fastapi import APIRouter


async def example_view():
    return {{"message": "This is an example view"}}
"""

TASKS_TEMPLATE = """
# {app_name}/tasks.py
from celery import shared_task


@shared_task
def example_task():
    return "This is an example task"
"""


@click.group()
def cli():
    pass


@click.command()
@click.argument('app_name')
def startapp(app_name):
    """Create a new app with the given name."""
    app_dir = os.path.join('apps', app_name)
    os.makedirs(app_dir)

    files = {
        '__init__.py': APP_TEMPLATE.format(app_name=app_name),
        'models.py': MODELS_TEMPLATE.format(app_name=app_name),
        'routers.py': ROUTERS_TEMPLATE.format(app_name=app_name),
        'views.py': VIEWS_TEMPLATE.format(app_name=app_name),
        'tasks.py': TASKS_TEMPLATE.format(app_name=app_name)
    }

    for filename, content in files.items():
        with open(os.path.join(app_dir, filename), 'w') as f:
            f.write(content)

    click.echo(f"App '{app_name}' created successfully!")


@click.command()
@click.option('--host', help='The host to bind to.')
@click.option('--port', default=8000, help='The port to bind to.')
def runserver(host, port):
    if DEBUG:
        host = host or '127.0.0.1'
        uvicorn.run("main:app", host=host, port=port, reload=True)
    else:
        host = host or '0.0.0.0'
        uvicorn.run("main:app", host=host, port=port, reload=False)


@click.command()
def runcelery():
    """Run the Celery worker."""
    celery_app.worker_main(["worker", "--loglevel=info"])


@click.command()
@click.argument('script_name')
def runscript(script_name):
    """Run a script with the FastAPI environment loaded."""
    import importlib.util

    # Initialize the FastAPI app and Tortoise ORM
    app = FastAPI()

    for app_name in INSTALLED_APPS:
        models_module = f"{app_name}.models"
        if models_module not in TORTOISE_ORM_CONFIG['apps']['models']['models']:
            TORTOISE_ORM_CONFIG['apps']['models']['models'].append(models_module)

    async def run():
        await init_db(db_config=TORTOISE_ORM_CONFIG)
        # Load and execute the script
        script_path = os.path.join(os.path.dirname(__file__), script_name)
        if not os.path.exists(script_path):
            click.echo(f"Script '{script_name}' not found.")
            sys.exit(1)

        spec = importlib.util.spec_from_file_location("script_module", script_path)
        script_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script_module)
        await script_module.main()
    run_async(run())


cli.add_command(startapp)
cli.add_command(runserver)
cli.add_command(runcelery)
cli.add_command(runscript)

if __name__ == '__main__':
    cli()
