from app.ml.wayang_detection import predict_wayang
from app.ml.model_registry import ModelRegistry

def detect_wayang(image, model_registry: ModelRegistry):
    try:
        detection_result = predict_wayang(image, model_registry)
        scores = detection_result["scores"]
        class_ids = detection_result["class_ids"]
        boxes = detection_result["boxes"]
        class_names = detection_result["class_names"]

        if len(scores) == 0:
            return {
                "error_message": "no detections found",
                "is_success": False,
                "prediction_type": None,
                "class_id": None,
                "class_name": None,
                "confidence": None,
                "class_scores": None,
                "detections": [],
            }

        detections = [
            {
                "class_id": int(class_ids[i]),
                "class_name": class_names[i],
                "confidence": float(scores[i]),
                "box": boxes[i].tolist(),
            }
            for i in range(len(scores))
        ]

        return {
            "error_message": None,
            "is_success": True,
            "prediction_type": "detection",
            "class_id": None,
            "class_name": None,
            "confidence": None,
            "class_scores": None,
            "detections": detections,
        }
    
    except Exception as e:
        return {
            "error_message": str(e),
            "is_success": False,
            "prediction_type": None,
            "class_id": None,
            "class_name": None,
            "confidence": None,
            "class_scores": None,
            "detections": [],
        }