import numpy as np
import os
import PIL
from numpy.lib.function_base import extract
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

# Set this to ignore any CLI warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Load dataset from Git repo
# DATA_DIRECTORY = "https://github.com/cruzalacar/Tf-DashSymbols/raw/master/tf_files/symbols-download.zip"
# DATA_DIRECTORY = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

# data_dir = tf.keras.utils.get_file(origin=DATA_DIRECTORY, 
#             fname='flowers_photos', 
#             extract=True)
# data_dir = pathlib.Path(data_dir)

import pathlib
dataset_url = "https://github.com/cruzalacar/Tf-DashSymbols/raw/master/tf_files/symbols.tar.gz"
data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                   fname='car_symbols',
                                   untar=True)
data_dir = pathlib.Path(data_dir)


# Verify image count etc
image_count = len(list(data_dir.glob('*/*.jpg'))) + len(list(data_dir.glob('*/*.png'))) + len(list(data_dir.glob('*/*.gif')))

print(f"Location of dataset: {data_dir}")
print(f"Total number of images in dataset: {image_count}")

if (image_count == 0):
    print("Error when training dataset. Data image count is 0")
    exit()

print("Caching complete. Will begin training process.")

# Create dataset

batch_size = 32
img_height = 180
img_width = 180

# Create the training model
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names

# Create the testing model
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


print(f"The following classes are being recognized: {train_ds.class_names}")

num_classes = 19

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)

model.summary()
model.save("./trained_model")


# image = tf.keras.preprocessing.image.load_img("predict_symbol.png", target_size=(img_height, img_width))
# img_array = tf.keras.utils.img_to_array(image)
# img_array = tf.expand_dims(img_array, 0) # Create a batch

# predictions = model.predict(img_array)
# score = tf.nn.softmax(predictions[0])

# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )
