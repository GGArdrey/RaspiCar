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
from tensorflow.keras.utils import to_categorical
import numpy as np

class PilotNetClassification:
    def __init__(self, data_dirs, save_dir):
        self.data_dirs = data_dirs
        date_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
        self.log_dir_base = save_dir + date_time + "/log"
        self.checkpoint_dir = save_dir + date_time + "/checkpoints"

    def map_label_to_class(self, label):
        boundaries = [-1.0, -0.66, -0.33, 0, 0.33, 0.66, 1.0]
        for i, boundary in enumerate(boundaries):
            if label <= boundary:
                return i
        return len(boundaries) - 1

    def load_dataset(self):
        paths = []
        labels = []
        for dir in self.data_dirs:
            filenames = os.listdir(dir)
            for filename in filenames:
                if filename.endswith('.jpg'):
                    paths.append(os.path.join(dir, filename))
                    raw_label = float(filename.split('_')[1].replace('.jpg', ''))
                    labels.append(self.map_label_to_class(raw_label))
        return paths, labels

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



    def load_and_preprocess_images(self, image_paths):
        images = []
        for image_path in image_paths:
            image = cv2.imread(image_path)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            image = self.scale_and_crop_image(image)
            images.append(image)
        return np.array(images)

    def create_generators(self, batch_size, train_paths, train_labels, test_paths, test_labels):
        train_gen = ImageDataGenerator(
            rotation_range=15,
            shear_range=0.1,
            zoom_range=0.2,
            brightness_range=(0.8, 1.2),
            fill_mode='nearest'
        )
        test_gen = ImageDataGenerator()

        # Load and preprocess images
        train_images = self.load_and_preprocess_images(train_paths)
        test_images = self.load_and_preprocess_images(test_paths)

        # Create the generators
        train_generator = train_gen.flow(
            train_images,
            train_labels,
            batch_size=batch_size
        )
        test_generator = test_gen.flow(
            test_images,
            test_labels,
            batch_size=batch_size
        )

        return train_generator, test_generator

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
        output = Dense(7, activation='softmax')(x)

        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, batch_size=32, epochs=100):
        paths, labels = self.load_dataset()
        train_paths, test_paths, train_labels, test_labels = train_test_split(paths, labels, test_size=0.2,
                                                                              random_state=42)
        train_labels = to_categorical(train_labels, num_classes=7)
        test_labels = to_categorical(test_labels, num_classes=7)
        checkpoint_callback = self.create_model_checkpoint()
        train_generator, test_generator = self.create_generators(batch_size, train_paths, train_labels, test_paths,
                                                                 test_labels)
        model = self.build_model()
        tensorboard_callback = TensorBoard(log_dir=self.log_dir_base, histogram_freq=1)

        epochs = 100
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=test_generator,
            callbacks=[tensorboard_callback, checkpoint_callback]
        )


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



    pilot_net = PilotNetClassification(data_dirs=data_dirs, save_dir="/home/luca/raspicar/training/")
    pilot_net.train()
