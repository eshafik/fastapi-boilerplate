# main.py
from fastapi import FastAPI
from tortoise.contrib.fastapi import register_tortoise
from config.settings import TORTOISE_ORM, DEBUG, INSTALLED_APPS
from config.middleware import CustomMiddleware
import importlib

app = FastAPI(debug=DEBUG)

app.add_middleware(CustomMiddleware)


# Function to dynamically import routers and models
def dynamic_import(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name, None)


# Dynamically include app routers and collect models
for app_name in INSTALLED_APPS:
    router = dynamic_import(f"{app_name}.routers", "router")
    if router:
        # app.include_router(router, prefix=f"/{app_name.split('.')[-1]}")
        app.include_router(router)

    models_module = dynamic_import(f"{app_name}.models", "__name__")
    if models_module:
        TORTOISE_ORM["apps"]["models"]["models"].append(f"{app_name}.models")

# Register Tortoise ORM
register_tortoise(
    app,
    config=TORTOISE_ORM,
    generate_schemas=True,
    add_exception_handlers=True,
)


@app.on_event("startup")
async def startup_event():
    from config.tortoise import init_tortoise
    await init_tortoise()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
