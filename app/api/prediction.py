from fastapi import APIRouter, File, UploadFile, WebSocket, WebSocketDisconnect

from app.helpers.load_image import decode_base64_image, load_image_from_bytes
from app.services.wayang_service import detect_wayang
from app.services.ketupat_service import classify_ketupat

router = APIRouter()


@router.post("/detect-wayang")
async def detect_wayang_endpoint(file: UploadFile = File(...)):
    result = detect_wayang(file)
    return result


@router.post("/classify-ketupat")
async def detect_ketupat_endpoint(file: UploadFile = File(...)):
    result = classify_ketupat(file)
    return result


@router.websocket("/ws/detect-wayang")
async def websocket_detect_wayang(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            image_payload = data.get("image")

            if not image_payload:
                await websocket.send_json({"error": "missing image payload"})
                continue

            try:
                if isinstance(image_payload, str):
                    image_bytes = decode_base64_image(image_payload)
                else:
                    image_bytes = bytes(image_payload)

                image = load_image_from_bytes(image_bytes)
                result = detect_wayang(image)
                await websocket.send_json({"result": result})
            except Exception as exc:
                await websocket.send_json({"error": str(exc)})
    except WebSocketDisconnect:
        return