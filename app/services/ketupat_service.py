import numpy as np
import app.ml.ketupat_classification as ketupat_classification

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

def classify_ketupat(image):
    classification_result = ketupat_classification.predict_ketupat(image)

    if classification_result.get("type") == "classification":
        class_id = classification_result["class_id"]
        return {
            "error_message": None,
            "is_success": True,
            "prediction_type": "classification",
            "class_id": class_id,
            "class_name": KETUPAT_CLASSES[class_id] if 0 <= class_id < len(KETUPAT_CLASSES) else None,
            "confidence": classification_result["confidence"],
            "class_scores": classification_result["class_scores"],
            "detections": [],
        }

    scores = classification_result["scores"]
    class_ids = classification_result["class_ids"]
    boxes = classification_result["boxes"]

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