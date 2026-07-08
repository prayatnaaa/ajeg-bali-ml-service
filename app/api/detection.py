from fastapi import APIRouter, File, UploadFile

from app.services.wayang_detection_service import detect_wayang

router = APIRouter()


@router.post("/detect-wayang")
async def detect_wayang_endpoint(file: UploadFile = File(...)):
    result = detect_wayang(file)
    return result