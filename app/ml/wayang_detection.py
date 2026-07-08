from app.ml.loader import wayang_model
import tensorflow as tf

def predict_wayang(image):
    input_details = wayang_model.get_input_details()
    output_details = wayang_model.get_output_details()

    # Preprocess the image
    image = tf.image.resize(image, (224, 224))
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, tf.float32) / 255.0

    # Set the input tensor
    wayang_model.set_tensor(input_details[0]['index'], image)

    # Run inference
    wayang_model.invoke()

    # Get the output tensor
    output_data = wayang_model.get_tensor(output_details[0]['index'])
    prediction = tf.nn.softmax(output_data[0])

    return prediction