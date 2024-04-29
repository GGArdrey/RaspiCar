import os

import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from datetime import datetime
import numpy as np



class PilotNet:
    def __init__(self, data_dir, save_dir):
        self.data_dir = data_dir
        date_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
        self.log_dir_base = save_dir + date_time + "/log"
        self.checkpoint_dir = save_dir + date_time + "/checkpoints"

    def load_dataset(self):
        filenames = os.listdir(self.data_dir)
        paths = [os.path.join(self.data_dir, f) for f in filenames]
        labels = [float(f.split('_')[1].replace('.jpg', '')) for f in filenames]
        return paths, labels

    def scale_and_crop_image(self, image, target_width=200, target_height=66):
        height, width = image.shape[:2]
        scaling_factor = target_width / width
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        resized_image = cv2.resize(image, (new_width, new_height))
        if new_height > target_height:
            y_start = new_height - target_height
        else:
            y_start = 0
        cropped_image = resized_image[y_start:y_start + target_height, 0:target_width]
        return cropped_image

    def create_model_checkpoint(self):
        # Define the checkpoint path using the .keras extension
        checkpoint_path = os.path.join(self.checkpoint_dir, "cp-{epoch:04d}.keras")

        # Create a ModelCheckpoint callback that saves the full model
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=False,  # Change this to False to save models at each epoch regardless of validation loss
            save_weights_only=False,  # Save the full model, not just the weights
            verbose=1,
            save_freq='epoch'  # Save the model at every epoch
        )
        return checkpoint_callback


    def create_generators(self, batch_size, train_paths, train_labels, test_paths, test_labels):
        train_gen = ImageDataGenerator(
            rotation_range=15,
            shear_range=0.1,
            zoom_range=0.2,
            brightness_range=(0.8, 1.2),
            fill_mode='nearest'
        )
        test_gen = ImageDataGenerator()

        train_generator = train_gen.flow_from_dataframe(
            dataframe=pd.DataFrame({'filename': train_paths, 'label': train_labels}),
            x_col='filename',
            y_col='label',
            target_size=(66, 200),
            batch_size=batch_size,
            class_mode='raw'
        )

        test_generator = test_gen.flow_from_dataframe(
            dataframe=pd.DataFrame({'filename': test_paths, 'label': test_labels}),
            x_col='filename',
            y_col='label',
            target_size=(66, 200),
            batch_size=batch_size,
            class_mode='raw'
        )

        return train_generator, test_generator

    def build_model(self):
        input_layer = Input(shape=(66, 200, 3))
        x = tf.keras.layers.Lambda(lambda x: tf.image.per_image_standardization(x))(input_layer)
        x = Conv2D(12, (5, 5), strides=(2, 2), activation="elu")(x)
        x = Conv2D(18, (5, 5), strides=(2, 2), activation="elu")(x)
        x = Conv2D(24, (5, 5), strides=(2, 2), activation="elu")(x)
        x = Dropout(0.1)(x)
        x = Conv2D(64, (3, 3), activation="elu")(x)
        x = Flatten()(x)
        x = Dropout(0.1)(x)
        x = Dense(100, activation="elu")(x)
        x = Dense(50, activation="elu")(x)
        x = Dense(10, activation="elu")(x)
        output = Dense(1)(x)

        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss='mse', optimizer='adam')
        return model

    def build_model2(self):
        input_layer = Input(shape=(66, 200, 3))
        x = tf.keras.layers.Lambda(lambda x: tf.image.per_image_standardization(x))(input_layer)
        x = Conv2D(24, (5, 5), strides=(2, 2), activation="elu")(x)
        x = Conv2D(36, (5, 5), strides=(2, 2), activation="elu")(x)
        x = Conv2D(48, (5, 5), strides=(2, 2), activation="elu")(x)
        x = Conv2D(64, (3, 3), activation="elu")(x)
        x = Dropout(0.2)(x)
        x = Conv2D(64, (3, 3), activation="elu")(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        x = Dense(100, activation="elu")(x)
        x = Dense(50, activation="elu")(x)
        x = Dense(10, activation="elu")(x)
        output = Dense(1)(x)

        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss='mse', optimizer='adam')
        return model

    def train(self, batch_size=32, epochs=100):
        paths, labels = self.load_dataset()
        train_paths, test_paths, train_labels, test_labels = train_test_split(paths, labels, test_size=0.2, random_state=42)
        checkpoint_callback = self.create_model_checkpoint()
        train_generator, test_generator = self.create_generators(batch_size, train_paths, train_labels, test_paths, test_labels)
        model = self.build_model()
        tensorboard_callback = TensorBoard(log_dir=self.log_dir_base, histogram_freq=1)

        epochs = 100
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=test_generator,
            callbacks=[tensorboard_callback, checkpoint_callback],
            use_multiprocessing=True
        )


# Example usage
if __name__ == '__main__':
    pilot_net = PilotNet(data_dir='/home/luca/raspicar/data/',save_dir="/home/luca/raspicar/training/")
    pilot_net.train()
