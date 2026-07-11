from app.ml.ketupat_classification import predict_ketupat
from app.ml.model_registry import ModelRegistry

def classify_ketupat(image, model_registry: ModelRegistry):
    classification_result = predict_ketupat(image, model_registry)

    if classification_result.get("type") != "classification":
        return {
            "error_message": "Something went wrong during ketupat classification",
            "is_success": False,
            "prediction_type": None,
            "class_id": None,
            "class_name": None,
            "confidence": None,
            "class_scores": None,
            "detections": [],
        }
    class_id = classification_result["class_id"]

    return {
            "error_message": None,
            "is_success": True,
            "prediction_type": "classification",
            "class_id": class_id,
            "class_name": classification_result["class_name"],
            "confidence": classification_result["confidence"],
            "class_scores": classification_result["class_scores"],
            "detections": [],
        }