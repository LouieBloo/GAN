import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
from tensorflow import keras
import time

img_width=64
img_height=64
channel_count=3
latent_dimension=32

train_ds = keras.preprocessing.image_dataset_from_directory(
    directory="D:/Downloads/Images/69000-20221206T023428Z-001/69000",
    label_mode=None, image_size=(img_width, img_height), batch_size=32,
    shuffle=True, seed=123, validation_split=0.2,subset="training"
).map(lambda x: x / 255.0)

def change_inputs(images):
  #x = tf.image.resize(normalization_layer(images),[28, 28], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return images, images


validation_ds = keras.preprocessing.image_dataset_from_directory(
    directory="D:/Downloads/Images/69000-20221206T023428Z-001/69000",
    label_mode=None, image_size=(img_width, img_height), batch_size=32,
    shuffle=True, seed=123, validation_split=0.2,subset="validation"
).map(lambda x: x / 255.0)

normalized_ds = train_ds.map(change_inputs)
normalized_validation_ds = validation_ds.map(change_inputs)

#dataset_num = train_ds.as_numpy_iterator()
# img = keras.preprocessing.image.array_to_img(out[0])
# img.save("test.png")


encoder_input_layer = keras.Input(shape=(img_width,img_height,channel_count),dtype='float32', name="EncoderInput")
next_layer = keras.layers.Flatten()(encoder_input_layer)
next_layer = keras.layers.Dense(64*64*3,activation="relu")(next_layer)
encoder_output_layer = keras.layers.Dense(latent_dimension,activation="relu")(next_layer)

encoder_model = keras.Model(encoder_input_layer,encoder_output_layer, name='EncoderModel')
encoder_model.summary()

#decoder_input_layer = keras.Input(shape=(64,),dtype='float32',name="DecoderInput")(encoder_output_layer)
decoder_input_layer = keras.layers.Dense(latent_dimension,activation="relu")(encoder_output_layer)
next_layer = keras.layers.Dense(64*64*3,activation="relu")(decoder_input_layer)
decoder_output_layer = keras.layers.Reshape((64,64,3))(next_layer)

decoder_model = keras.Model(decoder_input_layer,decoder_output_layer, name='DecoderModel')
decoder_model.summary()

autoencoder = keras.Model(encoder_input_layer,decoder_output_layer,name="autoencoder")
autoencoder.summary()

#opt = keras.optimizers.Adam(learning_rate=0.001,decay=1e-6)

autoencoder.compile(optimizer="adam",loss="mse")
autoencoder.fit(normalized_ds,validation_data=normalized_validation_ds,epochs=20)

out = autoencoder.predict(validation_ds)

x = 0
for pic in out:
    img = keras.preprocessing.image.array_to_img(pic)
    img.save("./images/gen/%03d.png" % (x))
    x = x + 1

x = 0
for pic in validation_ds:
    for p in pic:
        img = keras.preprocessing.image.array_to_img(p)
        img.save("./images/val/%03d.png" % (x))
        x = x + 1