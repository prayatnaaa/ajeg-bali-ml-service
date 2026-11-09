import numpy as np
from starlette.datastructures import UploadFile
from app.helpers.load_image import load_image_from_upload
from app.ml.model_registry import ModelRegistry

WAYANG_CLASSES = [
    "Acintya", "Arjuna", "Bhatara Siwa", "Bima", "Delem",
    "Durga", "Duryodana", "Krisna", "Kunti", "Madri",
    "Merdah", "Nakula-Sadewa", "Sangut", "Sengkuni", "Tualen", "Yudhistira",
]

_INPUT_SIZE = 640.0  # Model input resolution


def predict_wayang(image, model_registry: ModelRegistry):
    import tensorflow as tf

    if isinstance(image, UploadFile):
        image = load_image_from_upload(image)

    interpreter = model_registry.get_model("wayang_interpreter")
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image = tf.image.resize(image, (640, 640))
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, tf.float32) / 255.0

    interpreter.set_tensor(input_details[0]["index"], image)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])
    raw_output  = np.asarray(output_data[0]).T          # (num_preds, 4 + num_classes)

    if raw_output.shape[1] < 5:
        raise ValueError(f"Unexpected model output shape: {raw_output.shape}")

    boxes_cxcywh = raw_output[:, :4]                   # [cx, cy, w, h] in pixel coords
    class_scores  = raw_output[:, 4:]
    class_ids     = np.argmax(class_scores, axis=1)
    scores        = np.max(class_scores, axis=1)

    # --- Confidence pre-filter ---
    mask       = scores > 0.4                           # Keep candidates above 40%
    boxes_cxcywh = boxes_cxcywh[mask]
    scores        = scores[mask]
    class_ids     = class_ids[mask]

    if len(scores) > 0:
        # --- Convert [cx, cy, w, h] → [y1, x1, y2, x2] (normalized) ---
        # tf.image.non_max_suppression REQUIRES corner format, not center format.
        # Passing [cx, cy, w, h] directly produces invalid boxes (y1 > y2) and
        # completely breaks IoU calculation, so NMS never suppresses anything.
        cx, cy, w, h = (boxes_cxcywh[:, i] for i in range(4))
        boxes_xyxy = np.stack([
            cy - h / 2,   # y1
            cx - w / 2,   # x1
            cy + h / 2,   # y2
            cx + w / 2,   # x2
        ], axis=1) / _INPUT_SIZE                        # normalize to [0, 1]
        boxes_xyxy = np.clip(boxes_xyxy, 0.0, 1.0)

        selected = tf.image.non_max_suppression(
            tf.convert_to_tensor(boxes_xyxy, dtype=tf.float32),
            tf.convert_to_tensor(scores, dtype=tf.float32),
            max_output_size=10,
            iou_threshold=0.45,                         # was 0.5 — slightly more aggressive
            score_threshold=0.4,                        # matches pre-filter above
        )
        idx = selected.numpy()
        boxes_cxcywh = boxes_cxcywh[idx]
        scores        = scores[idx]
        class_ids     = class_ids[idx]

    return {
        "boxes":       boxes_cxcywh,                   # still [cx, cy, w, h] — Android expects this
        "scores":      scores,
        "class_ids":   class_ids,
        "class_names": [
            WAYANG_CLASSES[int(cid)] if 0 <= int(cid) < len(WAYANG_CLASSES) else None
            for cid in class_ids
        ],
    }