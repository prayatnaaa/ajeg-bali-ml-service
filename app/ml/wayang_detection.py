import numpy as np
import tensorflow as tf
from starlette.datastructures import UploadFile
from app.helpers.load_image import load_image_from_upload

from app.ml.loader import wayang_model

def predict_wayang(image):
    if isinstance(image, UploadFile):
        image = load_image_from_upload(image)

    input_details = wayang_model.get_input_details()
    output_details = wayang_model.get_output_details()

    image = tf.image.resize(image, (640, 640))
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, tf.float32) / 255.0

    wayang_model.set_tensor(input_details[0]['index'], image)
    wayang_model.invoke()

    output_data = wayang_model.get_tensor(output_details[0]['index'])
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
    }