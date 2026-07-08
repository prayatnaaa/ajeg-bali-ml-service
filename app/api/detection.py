from fastapi import APIRouter, UploadFile
from app.services.wayang_detection_service import detect_wayang

router = APIRouter()

@router.post("/detect-wayang")
async def detect_wayang_endpoint(file: UploadFile):

    result = detect_wayang(file)

    return result