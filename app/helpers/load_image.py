from base64 import b64decode
from starlette.datastructures import UploadFile
import tensorflow as tf

def load_image_from_upload(image_file: UploadFile) -> tf.Tensor:
    image_bytes = image_file.file.read()
    image_file.file.seek(0)

    if not image_bytes:
        raise ValueError("Received an empty image upload")

    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    return image


def load_image_from_bytes(image_bytes: bytes) -> tf.Tensor:
    if not image_bytes:
        raise ValueError("Received an empty image payload")

    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    return image


def decode_base64_image(base64_string: str) -> bytes:
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]
    return b64decode(base64_string)