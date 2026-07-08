from starlette.datastructures import UploadFile
import tensorflow as tf

def load_image_from_upload(image_file: UploadFile) -> tf.Tensor:
    image_bytes = image_file.file.read()
    image_file.file.seek(0)

    if not image_bytes:
        raise ValueError("Received an empty image upload")

    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    return image