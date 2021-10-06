import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 32
img_height = 180
img_width = 180

PREDICTION_FIlE = "fuel.jpeg"

class_names = ['airbag_deactivated', 'airbag_indicator', 'antilock_brake', 'battery_charge', 'brake_fluid', 'door_ajar', 'engine_temperature', 'engine_warning', 'exhaust_fluid', 'fuel_filter', 'glow_plug', 'low_fuel', 'oil_pressure', 'seat_belt', 'security_alert', 'tire_pressure', 'traction_control', 'traction_control_malfunction', 'washer_fluid']


model = tf.keras.models.load_model(
    "./trained_model", custom_objects=None, compile=True, options=None
)

image = tf.keras.preprocessing.image.load_img(PREDICTION_FIlE, target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(image)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

print(model.predict_classes)
