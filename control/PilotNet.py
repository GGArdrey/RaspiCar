import os

import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Input, Layer, Normalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from datetime import datetime
import numpy as np





class PilotNet:
    def __init__(self, data_dirs, save_dir):
        self.target_width = 200
        self.target_height = 66
        self.batch_size = 64
        self.data_dirs = data_dirs
        date_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
        self.log_dir_base = os.path.join(save_dir, date_time, "log")
        self.checkpoint_dir = os.path.join(save_dir, date_time, "checkpoints")

    def resize_and_crop_image(self, image):
        height, width = image.shape[:2]
        scaling_factor = self.target_width / width
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        resized_image = cv2.resize(image, (new_width, new_height))
        if new_height > self.target_height:
            y_start = (new_height - self.target_height) // 2
        else:
            y_start = 0
        cropped_image = resized_image[y_start:y_start + self.target_height, 0:self.target_width]
        return cropped_image

    def load_images_and_labels(self):
        labels = []
        images = []
        for dir in self.data_dirs:
            filenames = os.listdir(dir)
            for filename in filenames:
                if filename.endswith('.jpg'):
                    path = os.path.join(dir, filename)
                    image = cv2.imread(path)
                    if image is not None:
                        image = self.resize_and_crop_image(image)
                        images.append(image)
                        labels.append(float(filename.split('_')[1].replace('.jpg', '')))
        return np.array(images), np.array(labels)

    def augment_image(self, image, label):
        # Apply augmentation
        image = tf.image.random_brightness(image, max_delta=0.1)  # Random brightness
        return image, label

    def create_model_checkpoint(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, "cp-{epoch:04d}.keras")
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            save_freq='epoch'
        )
        return checkpoint_callback


    def build_model(self):
        norm_layer = Normalization()
        input_layer = Input(shape=(66, 200, 3))
        x = norm_layer(input_layer)
        x = Conv2D(24, (5, 5), strides=(2, 2), activation="elu")(x)
        x = Conv2D(36, (5, 5), strides=(2, 2), activation="elu")(x)
        x = Dropout(0.1)(x)
        x = Conv2D(48, (5, 5), strides=(2, 2), activation="elu")(x)
        x = Conv2D(64, (3, 3), activation="elu")(x)
        x = Flatten()(x)
        x = Dropout(0.1)(x)
        x = Dense(100, activation="elu")(x)
        x = Dense(50, activation="elu")(x)
        x = Dense(10, activation="elu")(x)
        output = Dense(1)(x)

        model = Model(inputs=input_layer, outputs=output)

        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # Compile the model with the defined optimizer
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def train(self, epochs=500):
        images, labels = self.load_images_and_labels()

        # Split data into train and validation sets
        train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2,
                                                                              random_state=42)
        print(f"Number of training samples: {len(train_images)}")
        print(f"Number of validation samples: {len(val_images)}")

        # Create training Dataset with augmentation
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_dataset = train_dataset.map(self.augment_image).shuffle(1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        # Create validation Dataset without augmentation
        val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        val_dataset = val_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        model = self.build_model()

        # Now, you can use this `dataset` directly in your model training
        tensorboard_callback = TensorBoard(log_dir=self.log_dir_base, histogram_freq=1)
        checkpoint_callback = self.create_model_checkpoint()

        model.fit(train_dataset, validation_data = val_dataset, epochs=epochs, callbacks=[tensorboard_callback,checkpoint_callback])



# Example usage
if __name__ == '__main__':
    # data_dirs = ["/home/luca/raspicar/data/29-04-2024_13-19-20",
    #              "/home/luca/raspicar/data/29-04-2024_13-20-08",
    #              "/home/luca/raspicar/data/29-04-2024_13-20-45",
    #              "/home/luca/raspicar/data/29-04-2024_13-21-21",
    #              "/home/luca/raspicar/data/29-04-2024_13-21-57",
    #              "/home/luca/raspicar/data/29-04-2024_13-22-33"]

    # data_dirs = ["/home/luca/raspicar/data/29-04-2024_15-16-45",
    #              "/home/luca/raspicar/data/29-04-2024_15-19-54",
    #              "/home/luca/raspicar/data/29-04-2024_15-22-37",
    #              "/home/luca/raspicar/data/29-04-2024_15-18-25",
    #              "/home/luca/raspicar/data/29-04-2024_15-21-09",
    #              "/home/luca/raspicar/data/29-04-2024_15-23-31"]

    data_dirs = ["/home/luca/raspicar/data/29-04-2024_15-16-45",
                 "/home/luca/raspicar/data/29-04-2024_15-19-54",
                 "/home/luca/raspicar/data/29-04-2024_15-22-37",
                 "/home/luca/raspicar/data/29-04-2024_15-18-25",
                 "/home/luca/raspicar/data/29-04-2024_15-21-09",
                 "/home/luca/raspicar/data/29-04-2024_15-23-31",
                 "/home/luca/raspicar/data/29-04-2024_15-33-14",
                 "/home/luca/raspicar/data/29-04-2024_15-33-54",
                 "/home/luca/raspicar/data/29-04-2024_15-34-44",
                 "/home/luca/raspicar/data/29-04-2024_15-35-14",
                 "/home/luca/raspicar/data/29-04-2024_15-35-48",
                 "/home/luca/raspicar/data/29-04-2024_15-36-14",
                 "/home/luca/raspicar/data/29-04-2024_15-36-56",
                 "/home/luca/raspicar/data/29-04-2024_15-37-34",
                 "/home/luca/raspicar/data/29-04-2024_15-38-15",
                 "/home/luca/raspicar/data/29-04-2024_15-33-31",
                 "/home/luca/raspicar/data/29-04-2024_15-34-18",
                 "/home/luca/raspicar/data/29-04-2024_15-34-58",
                 "/home/luca/raspicar/data/29-04-2024_15-35-36",
                 "/home/luca/raspicar/data/29-04-2024_15-36-00",
                 "/home/luca/raspicar/data/29-04-2024_15-36-32",
                 "/home/luca/raspicar/data/29-04-2024_15-37-14",
                 "/home/luca/raspicar/data/29-04-2024_15-37-49",
                 "/home/luca/raspicar/data/29-04-2024_16-22-36",
                 "/home/luca/raspicar/data/29-04-2024_16-23-27",
                 "/home/luca/raspicar/data/29-04-2024_16-24-28",
                 "/home/luca/raspicar/data/29-04-2024_16-25-28",
                 "/home/luca/raspicar/data/29-04-2024_16-26-16",
                 "/home/luca/raspicar/data/29-04-2024_16-27-14"
                 ]



    pilot_net = PilotNet(data_dirs=data_dirs, save_dir="/home/luca/raspicar/training/")
    pilot_net.train()
