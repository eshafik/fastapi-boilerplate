import logging
import asyncio

from fastapi import HTTPException
from tortoise import Tortoise, connections
from tortoise.exceptions import DBConnectionError

from config.settings import TORTOISE_ORM_CONFIG

logger = logging.getLogger("root")


async def init_db(db_config=TORTOISE_ORM_CONFIG):
    try:
        print('db initiated')
        await Tortoise.init(config=db_config)
        await Tortoise.generate_schemas()
        logger.info("✅ Tortoise ORM initialized")
    except DBConnectionError as e:
        logger.error(f"❌ Database connection failed: {e}")
        await reconnect_db()


async def close_db():
    await Tortoise.close_connections()
    logger.info("🔌 Database connections closed")


async def reconnect_db(retries=5, delay=5):
    for attempt in range(1, retries + 1):
        try:
            await close_db()
            await init_db()
            logger.info("🔄 Reconnected to DB")
            break
        except Exception as e:
            logger.error(f"⚠️ Retry {attempt}/{retries} failed: {e}")
            await asyncio.sleep(delay)
    else:
        logger.critical("💀 Could not reconnect to DB after retries")
        raise HTTPException(status_code=500, detail="Failed to reconnect to the database")
