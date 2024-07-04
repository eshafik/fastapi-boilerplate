# config/tortoise.py
from tortoise import Tortoise
from config.settings import TORTOISE_ORM

async def init_tortoise():
    await Tortoise.init(config=TORTOISE_ORM)
    await Tortoise.generate_schemas()
