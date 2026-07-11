import numpy as np
from starlette.datastructures import UploadFile
from app.helpers.load_image import load_image_from_upload
from app.ml.model_registry import ModelRegistry

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



def predict_wayang(image, model_registry: ModelRegistry):
    import tensorflow as tf

    if isinstance(image, UploadFile):
        image = load_image_from_upload(image)

    interpreter = model_registry.get_model("wayang_interpreter")
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image = tf.image.resize(image, (640, 640))
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, tf.float32) / 255.0

    interpreter.set_tensor(input_details[0]["index"], image)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])
    raw_output = np.asarray(output_data[0]).T

    if raw_output.shape[1] < 5:
        raise ValueError(f"Unexpected model output shape: {raw_output.shape}")

    boxes = raw_output[:, :4]
    class_scores = raw_output[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = np.max(class_scores, axis=1)

    mask = scores > 0.5
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(scores) > 0:
        selected = tf.image.non_max_suppression(
            tf.convert_to_tensor(boxes, dtype=tf.float32),
            tf.convert_to_tensor(scores, dtype=tf.float32),
            max_output_size=min(20, len(scores)),
            iou_threshold=0.5,
            score_threshold=0.5,
        )
        boxes = boxes[selected.numpy()]
        scores = scores[selected.numpy()]
        class_ids = class_ids[selected.numpy()]

    return {
        "boxes": boxes,
        "scores": scores,
        "class_ids": class_ids,
        "class_names": [
            WAYANG_CLASSES[int(class_id)]
            if 0 <= int(class_id) < len(WAYANG_CLASSES)
            else None
            for class_id in class_ids
        ]
    }