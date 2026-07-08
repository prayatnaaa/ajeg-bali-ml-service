from fastapi import APIRouter, File, UploadFile

from app.services.wayang_detection_service import detect_wayang

router = APIRouter()


@router.post("/detect-wayang")
async def detect_wayang_endpoint(file: UploadFile = File(...)):
    result = detect_wayang(file)
    return result

@router.post("/detect-ketupat")
async def detect_ketupat_endpoint(file: UploadFile = File(...)):
    from app.services.ketupat_detection_service import detect_ketupat

    result = detect_ketupat(file)
    return result