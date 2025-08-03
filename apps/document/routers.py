
# document/routers.py
from fastapi import APIRouter
from utils.response_wrapper import response_wrapper
from .views import (create_document, get_document, list_documents,
                    update_document, delete_document, get_document_chunks)

router = APIRouter()

router.post("/document/add")(response_wrapper(create_document))
router.get("/document/get")(response_wrapper(get_document))
router.get("/document/list")(response_wrapper(list_documents))
router.post("/document/update/{document_id}")(response_wrapper(update_document))
router.delete("/document/delete/{document_id}")(response_wrapper(delete_document))
router.get("/document/chunks/{document_id}")(response_wrapper(get_document_chunks))
