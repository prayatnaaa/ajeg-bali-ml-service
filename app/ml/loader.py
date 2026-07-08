import tensorflow as tf


wayang_model = tf.lite.Interpreter(
    model_path="../../dis_models/model_wayang.tflite"
)
wayang_model.allocate_tensors()

ketupat_model = tf.lite.Interpreter(
    model_path="../../dis_models/model_ketupat.tflite"
)
ketupat_model.allocate_tensors()