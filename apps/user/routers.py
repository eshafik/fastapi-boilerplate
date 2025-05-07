from fastapi import APIRouter
from utils.response_wrapper import response_wrapper
from .views import sign_up, get_token

router = APIRouter()

router.post("/api/v1/signup")(response_wrapper(sign_up))
router.post("/api/v1/token")(response_wrapper(get_token))
