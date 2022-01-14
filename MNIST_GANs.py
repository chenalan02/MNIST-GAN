import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from tensorflow.keras.layers import Input, Dense, ReLU, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0 * 2 - 1
x_test = x_test / 255.0 * 2 - 1

n, H, W = x_train.shape
D = H*W

x_train = x_train.reshape(-1, D)
x_test  = x_test.reshape(-1, D)

latent_dim = 100

i = Input(shape=(latent_dim,))
x = Dense(256, activation=LeakyReLU(alpha=0.2))(i)
x = BatchNormalization(momentum=0.7)(x)
x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
x = BatchNormalization(momentum=0.7)(x)
x = Dense(1024, activation=LeakyReLU(alpha=0.2))(x)
x = BatchNormalization(momentum=0.7)(x)
x = Dense(D, activation='tanh')(x)

generator = Model(i, x)

i = Input(shape=(D,))
x = Dense(512, activation=LeakyReLU(alpha=0.2))(i)
x = Dense(256, activation=LeakyReLU(alpha=0.2))(x)
x = Dense(1, activation='sigmoid')(x)

discriminator = Model(i, x)

discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(0.0002, 0.5),
    metrics=['accuracy'])

z = Input(shape=(latent_dim,))
img = generator(z)
discriminator.trainable = False
fake_pred = discriminator(img)
combined_model = Model(z, fake_pred)

combined_model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

batch_size = 32
epochs = 30000
sample_period = 200

ones = np.ones(batch_size)
zeros = np.zeros(batch_size)

if not os.path.exists('gan_images'):
    os.makedirs('gan_images')

def sample_images(epoch):
    rows, cols = 5, 5
    noise = np.random.randn(rows * cols, latent_dim)
    imgs = generator.predict(noise)

    # Rescale images 0 - 1
    imgs = 0.5 * imgs + 0.5

    fig, axs = plt.subplots(rows, cols)
    idx = 0
    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(imgs[idx].reshape(H, W), cmap='gray')
            axs[i, j].axis('off')
            idx += 1
    fig.savefig("gan_images/%d.png" % epoch)
    plt.close()

for epoch in range(epochs):

    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_imgs = x_train[idx]

    noise = np.random.randn(batch_size, latent_dim)
    fake_imgs = generator.predict(noise)

    # train discriminator
    discriminator.train_on_batch(real_imgs, ones)
    discriminator.train_on_batch(fake_imgs, zeros)

    # train generator
    noise = np.random.randn(batch_size, latent_dim)
    combined_model.train_on_batch(noise, ones)

    noise = np.random.randn(batch_size, latent_dim)
    combined_model.train_on_batch(noise, ones)

    if epoch % sample_period == 0:
        sample_images(epoch)
        print("epoch: "+str(epoch))