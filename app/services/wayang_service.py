import logging

import numpy as np

from app.ml.wayang_detection import predict_wayang

WAYANG_CLASSES = [
    "Acintya",
    "Arjuna",
    "Bhatara Siwa",
    "Bima",
    "Delem",
    "Durga",
    "Duryodana",
    "Krisna",
    "Kunti",
    "Madri",
    "Merdah",
    "Nakula-Sadewa",
    "Sangut",
    "Sengkuni",
    "Tualen",
    "Yudhistira",
]


def detect_wayang(image):
    detection_result = predict_wayang(image)
    scores = detection_result["scores"]
    class_ids = detection_result["class_ids"]
    boxes = detection_result["boxes"]

    if len(scores) == 0:
        return {
            "error_message": "no detections found",
            "is_success": False,
            "detections": [],
        }

    best_idx = int(np.argmax(scores))
    predicted_class = int(class_ids[best_idx])
    predicted_class_name = (
        WAYANG_CLASSES[predicted_class]
        if 0 <= predicted_class < len(WAYANG_CLASSES)
        else None
    )
    confidence = float(scores[best_idx])

    detections = [
        {
            "class_id": int(class_ids[i]),
            "class_name": (
                WAYANG_CLASSES[int(class_ids[i])]
                if 0 <= int(class_ids[i]) < len(WAYANG_CLASSES)
                else None
            ),
            "confidence": float(scores[i]),
            "box": boxes[i].tolist(),
        }
        for i in range(len(scores))
    ]

    logging.info(
        f"Predicted class: {predicted_class} ({predicted_class_name}), Confidence: {confidence}"
    )

    return {
        "error_message": None,
        "is_success": True,
        "detections": detections,
    }