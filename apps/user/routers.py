from fastapi import APIRouter
from utils.response_wrapper import response_wrapper
from .views import example_view

router = APIRouter()

router.get("/example")(response_wrapper(example_view))
