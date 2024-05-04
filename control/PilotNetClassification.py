import os

import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Input, Layer, Normalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from datetime import datetime
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.math import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import class_weight

def custom_loss(class_weights):
    class_weights = tf.constant(class_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        # Use built-in function to compute cross-entropy
        cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)

        # Get true and predicted classes as indices
        true_classes = tf.argmax(y_true, axis=1)
        pred_classes = tf.argmax(y_pred, axis=1)

        # Calculate class distances
        class_distance = tf.abs(tf.cast(true_classes, tf.float32) - tf.cast(pred_classes, tf.float32))
        distance_multiplier = 1.0 + (class_distance / (tf.cast(tf.shape(y_true)[1], tf.float32) - 1))

        # Apply class weights and distance multiplier
        weight_per_sample = tf.gather(class_weights, true_classes)
        weighted_loss = cross_entropy * tf.reshape(weight_per_sample * distance_multiplier, (-1, 1))

        return tf.reduce_mean(weighted_loss)

    return loss

class PilotNetClassification:
    def __init__(self, data_dirs, save_dir):
        self.target_width = 200
        self.target_height = 66
        self.batch_size = 64
        self.epochs = 1000
        self.data_dirs = data_dirs
        date_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
        self.log_dir_base = os.path.join(save_dir, date_time, "log/")
        if not os.path.exists(self.log_dir_base):
            os.makedirs(self.log_dir_base)
        self.checkpoint_dir = os.path.join(save_dir, date_time, "checkpoints")
        # self.boundaries = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
        #               0.8, 0.9, 1.0]

        self.boundaries = [-1.0,  -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6,  0.8, 1.0]
        self.flip_mapping = self.create_flip_mapping(len(self.boundaries))

    def map_label_to_class(self, label):
                # Calculate absolute differences between the label and each boundary
        differences = [abs(label - boundary) for boundary in self.boundaries]
        # Find the index of the smallest difference
        nearest_index = differences.index(min(differences))
        return nearest_index

    def load_images_and_labels(self):
        continuous_labels = []
        categorical_labels = []
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
                        angle = float(filename.split('_')[1].replace('.jpg', ''))
                        continuous_labels.append(angle)
                        categorical_labels.append(self.map_label_to_class(angle))
        return np.array(images), np.array(categorical_labels), np.array(continuous_labels)

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

    def resize_and_crop_image(self, image, target_width=200, target_height=66):
        # Check input image dimensions
        if len(image.shape) < 2:
            raise ValueError("Invalid image data!")

        height, width = image.shape[:2]

        # Calculate scaling factor to maintain aspect ratio based on width
        scaling_factor = target_width / width
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        resized_image = cv2.resize(image, (new_width, new_height))

        # Check if the new height is greater than or equal to the target height before cropping
        if new_height < target_height:
            raise ValueError("Resized image height is less than the target crop height.")

        # Calculate start y-coordinate for cropping to center the crop area
        y_start = new_height - target_height

        cropped_image = resized_image[y_start:y_start + target_height, 0:target_width]
        #cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
        return cropped_image

    def create_flip_mapping(self, num_classes):
        # This function creates a dictionary for flipping class indices
        return {i: len(self.boundaries) - 1 - i for i in range(len(self.boundaries))}

    def augment_image(self, image, label):
        # Randomly flip the image horizontally
        flip = tf.random.uniform([]) < 0.5  # 50% chance to flip
        image = tf.cond(flip, lambda: tf.image.flip_left_right(image), lambda: image)

        # Adjust label for flipped images
        # Convert one-hot to index
        label_index = tf.argmax(label, axis=-1)
        # Map indices if flipped
        flipped_index = tf.gather(tf.constant(list(self.flip_mapping.values())), label_index)
        # Convert index back to one-hot
        label = tf.cond(flip, lambda: tf.one_hot(flipped_index, depth=len(self.flip_mapping)), lambda: label)

        # Additional image augmentations
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_hue(image, max_delta=0.1)

        return image, label



    def build_model(self):
        norm_layer = Normalization()
        input_layer = Input(shape=(66, 200, 3))
        x = norm_layer(input_layer)
        x = Conv2D(24, (5, 5), strides=(2, 2), activation="elu")(x)
        x = Dropout(0.2)(x)
        x = Conv2D(36, (5, 5), strides=(2, 2), activation="elu")(x)
        x = Conv2D(48, (5, 5), strides=(2, 2), activation="elu")(x)
        x = Conv2D(64, (3, 3), activation="elu")(x)
        x = Dropout(0.2)(x)
        x = Conv2D(64, (3, 3), activation="elu")(x)
        x = Flatten()(x)
        x = Dense(100, activation="elu",kernel_regularizer=l2(0.01))(x)
        x = Dense(50, activation="elu",kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.2)(x)
        x = Dense(20, activation="elu",kernel_regularizer=l2(0.01))(x)
        output = Dense(len(self.boundaries), activation='softmax')(x)

        model = Model(inputs=input_layer, outputs=output)

        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        model.compile(loss=custom_loss([1.71521336, 3.23251748, 1.17000633, 1.04404291, 0.85469954, 0.39863457,
 1.01463325, 0.99462076, 0.99658642, 1.71521336, 0.96235253]), optimizer=optimizer, metrics=['accuracy'])
        return model

    def train(self, batch_size=32, epochs=100):
        images, labels, continuous_labels = self.load_images_and_labels()

        # Split data into train+val and test sets
        train_val_images, test_images, train_val_labels, test_labels = train_test_split(
            images, labels, test_size=0.1, random_state=42
        )

        # Split train+val into train and validation sets
        train_images, val_images, train_labels, val_labels = train_test_split(
            train_val_images, train_val_labels, test_size=0.2, random_state=42
        )

        class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                                  classes=np.arange(0, len(self.boundaries)),
                                                                  y=train_labels)
        print("Class Weights: ", class_weights)
        class_weights_dict = dict(zip(np.arange(0, len(self.boundaries)), class_weights))

        train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=len(self.boundaries)) #TODO hardcoded
        val_labels = tf.keras.utils.to_categorical(val_labels, num_classes=len(self.boundaries))
        test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=len(self.boundaries))

        print(f"Number of training samples: {len(train_images)}")
        print(f"Number of validation samples: {len(val_images)}")
        print(f"Number of testing samples: {len(test_images)}")
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_dataset = train_dataset.map(self.augment_image).shuffle(1000).batch(self.batch_size).prefetch(
            tf.data.AUTOTUNE)

        # Create validation Dataset without augmentation
        val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        val_dataset = val_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        # Create test Dataset without augmentation
        test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        test_dataset = test_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)



        model = self.build_model()

        tensorboard_callback = TensorBoard(log_dir=self.log_dir_base, histogram_freq=1)
        checkpoint_callback = self.create_model_checkpoint()

        self.write_log_pre_training(model, labels,continuous_labels, train_images, val_images, test_images, train_labels, val_labels,
                                    test_labels)

        model.fit(train_dataset, validation_data = val_dataset, epochs=self.epochs, callbacks=[tensorboard_callback,checkpoint_callback])
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16] #TODO maybe change optimization
        tflite_model = converter.convert()
        # Save the model.
        with open(self.checkpoint_dir + '/model.tflite', 'wb') as f:
            f.write(tflite_model)

        model.evaluate(test_dataset)


        test_predictions = model.predict(test_dataset)
        test_predictions = np.argmax(test_predictions, axis=1)
        true_labels = np.argmax(test_labels, axis=1)  # Adjust if test_labels are not in the correct shape

        # Generate the confusion matrix

        conf_matrix = confusion_matrix(true_labels, test_predictions)


        # Optionally, visualize the confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(len(self.boundaries)), yticklabels=range(len(self.boundaries)))
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.savefig(self.log_dir_base + "confusion_matrix")
        plt.show()

    def write_log_pre_training(self, model, labels, continuous_labels,train_images, val_images, test_images, train_labels, val_labels,
                               test_labels):
        '''
        Write a log before training with details to the data, data distribution plots and architecture
        :param model:
        :param train_images:
        :param val_images:
        :param test_images:
        :param train_labels:
        :param val_labels:
        :param test_labels:
        :return:
        '''
        with open(self.log_dir_base + "info.txt", 'a') as log:
            model.summary(print_fn=lambda x: log.write(x + '\n'))
            log.write(f"Epochs: {self.epochs}\n")
            log.write(f"Batchsize: {self.batch_size}\n\n")
            log.write(f"Data Sets Used: {self.data_dirs}\n\n")
            log.write(f"Classes used : {self.boundaries}\n\n")


            log.write("\nDistribution of Steering Angles for Complete Data\n")
            log.write(f"Number of Images: {len(labels)}\n")


            log.write("\nDistribution of Steering Angles for Training Data\n")
            log.write(f"Number of Images: {len(train_labels)}\n")


            log.write("\nDistribution of Steering Angles for Validation Data\n")
            log.write(f"Number of Images: {len(val_labels)}\n")

            log.write("\nDistribution of Steering Angles for Test Data\n")
            log.write(f"Number of Images: {len(test_labels)}\n")

            # Histogram for continuous labels
            plt.figure(figsize=(10, 5))
            plt.hist(continuous_labels, bins=20, color='blue', alpha=0.7)
            plt.title('Histogram of Continuous Steering Angles')
            plt.xlabel('Steering Angle')
            plt.ylabel('Frequency')
            plt.savefig(self.log_dir_base + "histogram_continuous.png")
            plt.close()

            # Histogram for categorical labels
            plt.figure(figsize=(10, 5))
            plt.hist(labels, bins=len(self.boundaries), color='green', alpha=0.7)
            plt.title('Histogram of Categorical Steering Angles')
            plt.xlabel('Steering Angle Categories')
            plt.ylabel('Frequency')
            plt.savefig(self.log_dir_base + "histogram_categorical.png")
            plt.close()


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
                 "/home/luca/raspicar/data/29-04-2024_16-27-14",
                 "/home/luca/raspicar/data/30-04-2024_13-44-21",
                 "/home/luca/raspicar/data/30-04-2024_13-45-32",
                 "/home/luca/raspicar/data/30-04-2024_13-47-07",
                 "/home/luca/raspicar/data/30-04-2024_13-48-12",
                 "/home/luca/raspicar/data/30-04-2024_13-49-53",
                 "/home/luca/raspicar/data/30-04-2024_13-50-31"]





    pilot_net = PilotNetClassification(data_dirs=data_dirs, save_dir="/home/luca/raspicar/training/")
    pilot_net.train()
