import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.preprocessing.image import array_to_img

# 1. Configure GPU for Dynamic Memory Growth


def configure_gpu():
    print("[INFO] Configuring GPU...")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# 2. Data Preparation with Label Conditioning


def load_dataset(batch_size=128):
    print("[INFO] Loading and preprocessing Fashion MNIST dataset...")
    (train_ds, _), ds_info = tfds.load('fashion_mnist',
                                       split=['train', 'test'],
                                       as_supervised=True,
                                       with_info=True)

    num_classes = ds_info.features['label'].num_classes

    def normalize_img(image, label):
        # Normalize pixel values and convert labels to one-hot
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.expand_dims(image, axis=-1)
        label = tf.one_hot(label, depth=num_classes)
        return image, label

    train_ds = train_ds.map(normalize_img)
    train_ds = train_ds.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, num_classes

# 3. Conditional Generator Model


def build_generator(latent_dim, num_classes):
    print("[INFO] Building Generator...")
    noise_input = layers.Input(shape=(latent_dim,))
    label_input = layers.Input(shape=(num_classes,))

    # Concatenate noise and label
    x = layers.Concatenate()([noise_input, label_input])
    x = layers.Dense(7 * 7 * 256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((7, 7, 256))(x)

    x = layers.Conv2DTranspose(128, 5, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(64, 5, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(1, 5, strides=2, padding='same', activation='sigmoid')(x)

    return Model([noise_input, label_input], x, name="Generator")

# 4. Conditional Discriminator Model


def build_discriminator(num_classes):
    print("[INFO] Building Discriminator...")
    image_input = layers.Input(shape=(28, 28, 1))
    label_input = layers.Input(shape=(num_classes,))

    # Expand label to match image shape
    label_embedding = layers.Dense(28 * 28)(label_input)
    label_embedding = layers.Reshape((28, 28, 1))(label_embedding)

    # Concatenate label with image
    x = layers.Concatenate()([image_input, label_embedding])

    x = layers.Conv2D(64, 5, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, 5, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    return Model([image_input, label_input], x, name="Discriminator")

# 5. Custom Callback to Save Images After Each Epoch


class ImageSaver(Callback):
    def __init__(self, generator, latent_dim, num_classes, save_dir='images'):
        self.generator = generator
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        print(f"[INFO] Saving generated images for epoch {epoch}...")
        random_noise = tf.random.normal((self.num_classes, self.latent_dim))
        labels = tf.eye(self.num_classes)  # One-hot for each class

        generated_images = self.generator([random_noise, labels])
        generated_images *= 255

        for i in range(self.num_classes):
            img = array_to_img(generated_images[i])
            img.save(os.path.join(self.save_dir, f"epoch_{epoch}_class_{i}.png"))

# 6. GAN Class Definition


class ConditionalGAN(Model):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, g_opt, d_opt, g_loss_fn, d_loss_fn):
        super().compile()
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss_fn = g_loss_fn
        self.d_loss_fn = d_loss_fn

    def train_step(self, batch):
        real_images, real_labels = batch
        batch_size = tf.shape(real_images)[0]

        # Generate fake images
        random_noise = tf.random.normal((batch_size, 128))
        fake_images = self.generator([random_noise, real_labels])

        # Train Discriminator
        with tf.GradientTape() as tape:
            pred_real = self.discriminator([real_images, real_labels])
            pred_fake = self.discriminator([fake_images, real_labels])
            d_loss = self.d_loss_fn(tf.ones_like(pred_real), pred_real) + \
                     self.d_loss_fn(tf.zeros_like(pred_fake), pred_fake)

        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        # Train Generator
        random_noise = tf.random.normal((batch_size, 128))
        with tf.GradientTape() as tape:
            generated_images = self.generator([random_noise, real_labels])
            predictions = self.discriminator([generated_images, real_labels])
            g_loss = self.g_loss_fn(tf.ones_like(predictions), predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(grads, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}

# 7. Main Function


def main():
    configure_gpu()

    # Load dataset and number of classes
    dataset, num_classes = load_dataset()

    # Build generator and discriminator
    generator = build_generator(latent_dim=128, num_classes=num_classes)
    discriminator = build_discriminator(num_classes=num_classes)

    # Setup losses and optimizers
    g_loss_fn = BinaryCrossentropy()
    d_loss_fn = BinaryCrossentropy()
    g_opt = Adam(1e-4)
    d_opt = Adam(1e-4)

    # Create and compile GAN model
    gan = ConditionalGAN(generator, discriminator)
    gan.compile(g_opt, d_opt, g_loss_fn, d_loss_fn)

    # Setup callbacks
    saver = ImageSaver(generator, latent_dim=128, num_classes=num_classes)
    checkpoint = ModelCheckpoint("generator_best.h5", save_best_only=True, monitor='g_loss', mode='min', save_weights_only=True)

    print("[INFO] Starting GAN training...")
    history = gan.fit(dataset, epochs=50, callbacks=[saver, checkpoint])

    # Plot losses after training
    print("[INFO] Plotting loss curves...")
    plt.plot(history.history['g_loss'], label='Generator Loss')
    plt.plot(history.history['d_loss'], label='Discriminator Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("GAN Training Losses")
    plt.savefig("images/loss_plot.png")
    plt.show()

    print("[INFO] Training Complete. Models and images saved.")


# Run Script
if __name__ == "__main__":
    main()
