import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt



dataset = keras.preprocessing.image_dataset_from_directory(
    directory="D:/Downloads/Images/69000-20221206T023428Z-001/69000",
    label_mode=None, image_size=(64, 64), batch_size=32,
    shuffle=True, seed=None, validation_split=None,
).map(lambda x: x / 255.0)

discriminator = keras.Sequential(
    [
        keras.Input(shape=(64, 64, 3)),
        layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator",
)
print(discriminator.summary())

latent_dim = 128
generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        layers.Dense(8 * 8 * 128),
        layers.Reshape((8, 8, 128)),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
    ],
    name="generator",
)
generator.summary()

opt_gen = keras.optimizers.Adam(1e-4)
opt_disc = keras.optimizers.Adam(1e-4)
loss_fn = keras.losses.BinaryCrossentropy()

x = 0
havent_saved_pic = True
for epoch in range(1000):
    havent_saved_pic = True
    for idx, (real) in enumerate(tqdm(dataset)):
        batch_size = real.shape[0]
        with tf.GradientTape() as gen_tape:
            random_latent_vectors = tf.random.normal(shape = (batch_size, latent_dim))
            fake = generator(random_latent_vectors)

        if x % 10 == 0 and havent_saved_pic:
            img = keras.preprocessing.image.array_to_img(fake[0])
            img.save("generated_img_%03d_%d.png" % (epoch, idx))
            havent_saved_pic = False

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        with tf.GradientTape() as disc_tape:
            loss_disc_real = loss_fn(tf.ones((batch_size, 1)), discriminator(real))
            loss_disc_fake = loss_fn(tf.zeros((batch_size, 1)), discriminator(fake))
            loss_disc = (loss_disc_real + loss_disc_fake)/2

        grads = disc_tape.gradient(loss_disc, discriminator.trainable_weights)
        opt_disc.apply_gradients(
            zip(grads, discriminator.trainable_weights)
        )

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        with tf.GradientTape() as gen_tape:
            fake = generator(random_latent_vectors)
            output = discriminator(fake)
            loss_gen = loss_fn(tf.ones(batch_size, 1), output)

        grads = gen_tape.gradient(loss_gen, generator.trainable_weights)
        opt_gen.apply_gradients(zip(grads, generator.trainable_weights))

    x = x + 1


last_pic_noise = tf.random.normal(shape = (1, latent_dim))
last_pic = generator(random_latent_vectors, training=False)
img = keras.preprocessing.image.array_to_img(last_pic[0])
img.save("last.png")