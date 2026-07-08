from app.ml.wayang_detection import predict_wayang
import tensorflow as tf

def detect_wayang(image):
    prediction = predict_wayang(image)
    predicted_class = tf.argmax(prediction).numpy()
    confidence = tf.reduce_max(prediction).numpy()

    return predicted_class, confidence