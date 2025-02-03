from fastapi import APIRouter
from backend.mcq_api import router as mcq_router
from backend.document_api import router as document_router

router = APIRouter()

router.include_router(mcq_router, prefix="/mcq", tags=["MCQ Generation"])
router.include_router(document_router, prefix="/document", tags=["Document Processing"])

