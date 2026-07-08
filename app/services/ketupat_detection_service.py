import numpy as np
import app.ml.ketupat_detection as ketupat_detection

KETUPAT_CLASSES = [
    "Bagia",
    "Dampulan",
    "Gatep",
    "Gong",
    "Kukur",
    "Lepet",
    "Lojor",
    "Nasi",
    "Sari",
    "Sidakarya",
    "Sidapurna",
    "Sirikan",
    "Taluh",
]

def detect_ketupat(image):
    detection_result = ketupat_detection.predict_ketupat(image)

    if detection_result.get("type") == "classification":
        class_id = detection_result["class_id"]
        return {
            "error_message": None,
            "is_success": True,
            "prediction_type": "classification",
            "class_id": class_id,
            "class_name": KETUPAT_CLASSES[class_id] if 0 <= class_id < len(KETUPAT_CLASSES) else None,
            "confidence": detection_result["confidence"],
            "class_scores": detection_result["class_scores"],
            "detections": [],
        }

    scores = detection_result["scores"]
    class_ids = detection_result["class_ids"]
    boxes = detection_result["boxes"]

    if len(scores) == 0:
        return {
            "error_message": "no detections found",
            "is_success": False,
            "detections": [],
        }

    detections = [
        {
            "class_id": int(class_ids[i]),
            "confidence": float(scores[i]),
            "box": boxes[i].tolist(),
        }
        for i in range(len(scores))
    ]

    return {
        "error_message": None,
        "is_success": True,
        "prediction_type": "detection",
        "detections": detections,
    }