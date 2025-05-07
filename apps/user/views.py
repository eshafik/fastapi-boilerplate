from fastapi import HTTPException

from apps.user.schema import UserCreate, UserCred
from apps.user.services import create_user, authenticate_user
from utils.jwt import generate_jwt_token


async def sign_up(data: UserCreate):
    data = data.model_dump()
    user = await create_user(**data)
    token = generate_jwt_token(user=user)
    return {'token': token}


async def get_token(data: UserCred):
    data = data.model_dump()
    user = await authenticate_user(**data)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = generate_jwt_token(user=user)
    return {'token': token}
