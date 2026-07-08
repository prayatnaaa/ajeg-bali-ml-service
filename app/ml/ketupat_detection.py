import numpy as np
import tensorflow as tf
from starlette.datastructures import UploadFile

from app.helpers.load_image import load_image_from_upload
from app.ml.loader import ketupat_model


def predict_ketupat(image):
    if isinstance(image, UploadFile):
        image = load_image_from_upload(image)

    input_details = ketupat_model.get_input_details()
    output_details = ketupat_model.get_output_details()

    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")

    image = tf.image.resize(image, (224, 224))
    image = tf.expand_dims(image, axis=0)
    # image = tf.cast(image, tf.float32) / 255.0
    image = (tf.cast(image, tf.float32) / 127.5) - 1.0

    ketupat_model.set_tensor(input_details[0]['index'], image)
    ketupat_model.invoke()

    output_data = ketupat_model.get_tensor(output_details[0]['index'])
    raw_output = np.asarray(output_data[0])

    print(f"Raw output shape: {raw_output.shape}")
    print(f"Raw output: {raw_output}")
    print("argmax", np.argmax(raw_output))
    print("top scores", np.sort(raw_output)[-5:])

    if raw_output.ndim == 1 or (raw_output.ndim == 2 and min(raw_output.shape) == 1):
        class_scores = raw_output.flatten()
        class_id = int(np.argmax(class_scores))
        confidence = float(np.max(class_scores))

        return {
            "type": "classification",
            "class_id": class_id,
            "confidence": confidence,
            "class_scores": class_scores.tolist(),
        }

    raw_output = raw_output.T

    if raw_output.ndim != 2 or raw_output.shape[1] < 5:
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
        "type": "detection",
        "boxes": boxes,
        "scores": scores,
        "class_ids": class_ids,
    }
