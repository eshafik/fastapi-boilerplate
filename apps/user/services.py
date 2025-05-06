import uuid

from apps.user.models import User
from utils.security import hash_password
from typing import Optional, Tuple


async def create_user(email: Optional[str] = None,
                      username: Optional[str] = None,
                      password: Optional[str] = None,
                      name: Optional[str] = None) -> User:
    hashed_pwd = password and hash_password(password) or None
    username = username or uuid.uuid4().hex
    user = await User.create(username=username, email=email, password=hashed_pwd, name=name)
    return user


async def get_user_by_email(email: str) -> Optional[User]:
    user = await User.filter(email=email).first()
    return user


async def get_user_by_username(username: str) -> Optional[User]:
    user = await User.filter(username=username).first()
    return user


async def get_or_create_user(email: str = None,
                             username: str = None,
                             name: str = None,
                             password: str = None) -> Tuple[bool, User]:
    user = None
    if email:
        user = await get_user_by_email(email=email)
    elif username:
        user = await get_user_by_username(username=username)
    if user:
        return False, user
    user = await create_user(email=email, username=username, name=name, password=password)
    return True, user
