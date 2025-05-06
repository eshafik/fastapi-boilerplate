import asyncio
import importlib

import uvloop
from fastapi import FastAPI, Request, HTTPException
from tortoise import connections
from tortoise.exceptions import DBConnectionError

from config.db import init_db, close_db, reconnect_db
from config.settings import TORTOISE_ORM_CONFIG, DEBUG, INSTALLED_APPS
from config.middleware import CustomMiddleware

uvloop.install()

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
        TORTOISE_ORM_CONFIG["apps"]["models"]["models"].append(f"{app_name}.models")

# # Register Tortoise ORM
# register_tortoise(
#     app,
#     config=TORTOISE_ORM,
#     generate_schemas=True,
#     add_exception_handlers=True,
# )


@app.on_event("startup")
async def startup_event():
    await init_db()
    print(f"Using event loop: {type(asyncio.get_event_loop())}")


@app.on_event("shutdown")
async def shutdown_event():
    await close_db()


@app.middleware("http")
async def db_session_middleware(request: Request, call_next):
    try:
        # Opens connection if not already
        connection = connections.get("default")
        if not connection.is_connected:
            await connection.connect()

        response = await call_next(request)
        return response
    except DBConnectionError as db_err:
        await reconnect_db()
        raise HTTPException(status_code=500, detail="Database error occurred")
    finally:
        # Optional: don't close if persistent connection is preferred
        # await connections.get("default").close()
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
