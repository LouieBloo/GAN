
# Check adding noise to input (we already did this)
# Discriminator is probably too good
# Maybe back off the input noise

import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
from tensorflow import keras
import time

img_height=64
img_width=64

def make_generator_model():
    model = tf.keras.Sequential(name="generator")
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(128,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 256)))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid'))

    return model


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def make_discriminator_model():
    model = tf.keras.Sequential(name="discrimiator")
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[img_height, img_width, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = (real_loss + fake_loss)
    return total_loss



# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
# @tf.function
def train_step(images,epoch):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      #real_output = discriminator(images + tf.random.normal(shape=tf.shape(images), mean=0.0, stddev=0.1, dtype=tf.float32), training=True)
      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


    return gen_loss, disc_loss
    

def train(dataset, epochs):
  for epoch in range(epochs):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    start = time.time()

    for image_batch in dataset:
      gen_loss, disc_loss = train_step(image_batch,epoch)


    print(gen_loss)
    print(disc_loss)
    # Produce images for the GIF as you go
    #display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             noise)

    # Save the model every 15 epochs
    #if (epoch + 1) % 15 == 0:
      #checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  #display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           noise)


def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  # fig = plt.figure(figsize=(4, 4))

  # for i in range(predictions.shape[0]):
  #     plt.subplot(4, 4, i+1)
  #     print(predictions[i])
  #     plt.imshow(predictions[i])
  #     plt.axis('off')

  # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  img = keras.preprocessing.image.array_to_img(predictions[0])
  img.save('./images/image_at_epoch_{:04d}.png'.format(epoch))
  #plt.show()





# (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
# train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
# train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
# Batch and shuffle the data
#train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


BUFFER_SIZE = 60000
BATCH_SIZE = 8

train_ds = tf.keras.utils.image_dataset_from_directory(
  "D:/Downloads/Images/69000-20221206T023428Z-001/69000",
#   validation_split=0.2,
#   subset="training",
  labels=None,
  label_mode=None,
  seed=None,
  image_size=(img_height, img_width),
  batch_size=BATCH_SIZE,
 )
  

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x: (normalization_layer(x)))


generator = make_generator_model()
generator.summary()
discriminator = make_discriminator_model()
discriminator.summary()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

EPOCHS = 2000
noise_dim = 128
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])

# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator)



# generator = make_generator_model()
# generator.summary()
# noise = tf.random.normal([1, 128])
# generated_image = generator(noise, training=False)

# print(generated_image)
# plt.imshow(generated_image[0])
# #plt.imshow( tf.random.uniform(shape=(128,128,3),minval=0, maxval=1, dtype=tf.float32))
# plt.show()

train(train_ds, EPOCHS)



# img = keras.preprocessing.image.array_to_img(generated_image[0])
# img.save("last22.png")