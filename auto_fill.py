import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
from tensorflow import keras
import time
from PIL import Image

img_width=64
img_height=64
channel_count=3
latent_dimension=512
batch_size=32
epochs = 10
loss="mse"
#loss="mae"
learning_rate=0.0001
decay=0
#decay=1e-6

#"D:/Downloads/Images/69000-20221206T023428Z-001/69000"
holed_cat_path="D:\Downloads\Images\cat_holed"
valid_cat_path="D:\Downloads\Images\dataset-part1"

train_holed = keras.preprocessing.image_dataset_from_directory(
    directory=holed_cat_path,
    label_mode=None, image_size=(img_width, img_height), batch_size=batch_size,
    shuffle=False, seed=123, validation_split=0.1,subset="training"
).map(lambda x: x / 255.0)
validation_holed = keras.preprocessing.image_dataset_from_directory(
    directory=holed_cat_path,
    label_mode=None, image_size=(img_width, img_height), batch_size=batch_size,
    shuffle=False, seed=123, validation_split=0.1,subset="validation"
).map(lambda x: x / 255.0)

def change_inputs(holed_images, valid_images):
  #x = tf.image.resize(normalization_layer(images),[28, 28], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return images, images

train_source = keras.preprocessing.image_dataset_from_directory(
    directory=valid_cat_path,
    label_mode=None, image_size=(img_width, img_height), batch_size=batch_size,
    shuffle=False, seed=123, validation_split=0.1,subset="training"
).map(lambda x: x / 255.0)
validation_source = keras.preprocessing.image_dataset_from_directory(
    directory=valid_cat_path,
    label_mode=None, image_size=(img_width, img_height), batch_size=batch_size,
    shuffle=False, seed=123, validation_split=0.1,subset="validation"
).map(lambda x: x / 255.0)


train_ds = tf.data.Dataset.zip((train_holed,train_source))
validation_ds = tf.data.Dataset.zip((validation_holed,validation_source))

# normalized_ds = train_ds.map(change_inputs)
# normalized_validation_ds = validation_ds.map(change_inputs)

#dataset_num = train_ds.as_numpy_iterator()
# img = keras.preprocessing.image.array_to_img(out[0])
# img.save("test.png")


encoder_input_layer = keras.Input(shape=(img_width,img_height,channel_count),dtype='float32', name="EncoderInput")
next_layer = keras.layers.Conv2D(32,3,padding='same',activation="relu")(encoder_input_layer)
next_layer = keras.layers.MaxPooling2D(pool_size=(2,2),padding='valid')(next_layer)
next_layer = keras.layers.Conv2D(64,3,padding='same',activation="relu")(next_layer)
next_layer = keras.layers.Conv2D(64,3,padding='same',activation="relu")(next_layer)
next_layer = keras.layers.Flatten()(next_layer)
encoder_output_layer = keras.layers.Dense(latent_dimension,activation="sigmoid")(next_layer)

encoder_model = keras.Model(encoder_input_layer,encoder_output_layer, name='EncoderModel')
encoder_model.summary()

#decoder_input_layer = keras.Input(shape=(64,),dtype='float32',name="DecoderInput")(encoder_output_layer)
decoder_input_layer = keras.layers.Dense(latent_dimension,activation="relu")(encoder_output_layer)
next_layer = keras.layers.Dense(img_width*img_height*3,activation="relu")(decoder_input_layer)
next_layer = keras.layers.Reshape((img_width,img_height,3))(next_layer)
next_layer = keras.layers.Conv2DTranspose(64,3,padding='same',activation="relu")(next_layer)
next_layer = keras.layers.Conv2DTranspose(32,3,padding='same',activation="relu")(next_layer)
decoder_output_layer = keras.layers.Conv2D(3,3,padding='same',activation="sigmoid")(next_layer)

decoder_model = keras.Model(decoder_input_layer,decoder_output_layer, name='DecoderModel')
decoder_model.summary()

autoencoder = keras.Model(encoder_input_layer,decoder_output_layer,name="autoencoder")
autoencoder.summary()

opt = keras.optimizers.Adam(learning_rate=learning_rate,decay=decay)
#opt = keras.optimizers.Adam(1e-4)

autoencoder.compile(opt,loss=loss)
autoencoder.fit(train_ds,validation_data=validation_ds,epochs=epochs)

out = autoencoder.predict(validation_holed)

x = 0
for pic in out:
    img = keras.preprocessing.image.array_to_img(pic)
    img.save("./images/gen/%03d.png" % (x))
    x = x + 1

# x = 0
# for pic in validation_holed:
#     for p in pic:
#         img = keras.preprocessing.image.array_to_img(p)
#         img.save("./images/val/%03d.png" % (x))
#         x = x + 1