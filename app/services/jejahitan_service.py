from starlette.datastructures import UploadFile
from app.ml.jejahitan_classification import predict_jejahitan
from app.ml.model_registry import ModelRegistry


def classify_jejahitan(image: UploadFile, model_registry: ModelRegistry):
    classification_result = predict_jejahitan(image, model_registry)
    return {
        "error_message": None,
        "is_success": True,
        "prediction_type": "classification",
        "class_id": classification_result["class_id"],
        "class_name": classification_result["class_name"],
        "confidence": classification_result["confidence"],
        "class_scores": classification_result["class_scores"],
        "detections": [],
    }
