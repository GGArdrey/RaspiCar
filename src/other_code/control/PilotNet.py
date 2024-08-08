import os
import sys

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
        self.epochs = 25
        self.data_dirs = data_dirs
        date_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
        self.log_dir_base = os.path.join(save_dir, date_time, "log/")
        if not os.path.exists(self.log_dir_base):
            os.makedirs(self.log_dir_base)
        self.checkpoint_dir = os.path.join(save_dir, date_time, "checkpoints")

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
            save_best_only=False,
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
        x = Dropout(0.2)(x)
        x = Conv2D(36, (5, 5), strides=(2, 2), activation="elu")(x)
        x = Conv2D(48, (5, 5), strides=(2, 2), activation="elu")(x)
        x = Conv2D(64, (3, 3), activation="elu")(x)
        x = Dropout(0.2)(x)
        x = Conv2D(64, (3, 3), activation="elu")(x)
        x = Flatten()(x)
        x = Dense(100, activation="elu")(x)
        x = Dense(50, activation="elu")(x)
        x = Dropout(0.2)(x)
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
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
        return model

    def train(self):
        images, labels = self.load_images_and_labels()

        # Split data into train+val and test sets
        train_val_images, test_images, train_val_labels, test_labels = train_test_split(
            images, labels, test_size=0.1, random_state=42
        )

        # Split train+val into train and validation sets
        train_images, val_images, train_labels, val_labels = train_test_split(
            train_val_images, train_val_labels, test_size=0.2, random_state=42
        )

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

        # Now, you can use this `dataset` directly in your model training
        tensorboard_callback = TensorBoard(log_dir=self.log_dir_base, histogram_freq=1)
        checkpoint_callback = self.create_model_checkpoint()

        self.write_log_pre_training(model, labels, train_images,val_images, test_images,train_labels,val_labels,test_labels)

        model.fit(train_dataset, validation_data = val_dataset, epochs=self.epochs, callbacks=[tensorboard_callback,checkpoint_callback])

        # Now convert the last/best model to tf lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16] #TODO maybe change optimization
        tflite_model = converter.convert()
        # Save the model.
        with open(self.checkpoint_dir + '/model.tflite', 'wb') as f:
            f.write(tflite_model)

        results = model.evaluate(test_dataset)
        print(f"Test MSE: {results[0]}, Test MAE: {results[1]}, Test RMSE: {results[2]}")
        self.write_log_post_training(model, test_dataset, test_images, test_labels)



    def write_log_pre_training(self, model, labels, train_images, val_images, test_images, train_labels, val_labels, test_labels):
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
        with open(self.log_dir_base+"info.txt", 'a') as log:
            model.summary(print_fn=lambda x: log.write(x + '\n'))
            log.write(f"Epochs: {self.epochs}\n")
            log.write(f"Batchsize: {self.batch_size}\n\n")
            log.write(f"Data Sets Used: {self.data_dirs}\n\n")


            mean_angle, variance_angle = self.create_data_distribution_graph(labels, "Distribution of Steering Angles for Complete Data")
            log.write("\nDistribution of Steering Angles for Complete Data\n")
            log.write(f"Number of Images: {len(labels)}\n")
            log.write(f"Mean of Steering Angles: {mean_angle:.2f}\n")
            log.write(f"Variance of Steering Angles: {variance_angle:.2f}\n")


            mean_angle, variance_angle = self.create_data_distribution_graph(train_labels,"Distribution of Steering Angles for Training Data")
            log.write("\nDistribution of Steering Angles for Training Data\n")
            log.write(f"Number of Images: {len(train_labels)}\n")
            log.write(f"Mean of Steering Angles: {mean_angle:.2f}\n")
            log.write(f"Variance of Steering Angles: {variance_angle:.2f}\n")


            mean_angle, variance_angle = self.create_data_distribution_graph(val_labels, "Distribution of Steering Angles for Validation Data")
            log.write("\nDistribution of Steering Angles for Validation Data\n")
            log.write(f"Number of Images: {len(val_labels)}\n")
            log.write(f"Mean of Steering Angles: {mean_angle:.2f}\n")
            log.write(f"Variance of Steering Angles: {variance_angle:.2f}\n")


            mean_angle, variance_angle = self.create_data_distribution_graph(test_labels, "Distribution of Steering Angles for Test Data")
            log.write("\nDistribution of Steering Angles for Test Data\n")
            log.write(f"Number of Images: {len(test_labels)}\n")
            log.write(f"Mean of Steering Angles: {mean_angle:.2f}\n")
            log.write(f"Variance of Steering Angles: {variance_angle:.2f}\n")

    def create_data_distribution_graph(self, labels, title='Distribution of Steering Angles',bins=21):
        import matplotlib.pyplot as plt
        """
        Plots the distribution of steering angles.

        Args:
        - angles (list or numpy array): The list or array containing the steering angles.
        - bins (int): Number of bins in the histogram.

        """

        # Calculating statistics
        mean_angle = np.mean(labels)
        variance_angle = np.var(labels)

        # Printing statistics
        print("Number of Images: ", len(labels))
        print(f"Mean of Steering Angles: {mean_angle:.2f}")
        print(f"Variance of Steering Angles: {variance_angle:.2f}")

        plt.figure(figsize=(10, 6))
        counts, bin_edges, _ = plt.hist(labels, bins=bins, alpha=0.7, color='blue')
        plt.title(title)
        plt.xlabel('Steering Angle')
        plt.ylabel('Frequency')
        plt.grid(True)

        # Adding text labels above bars
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Calculate bin centers
        for count, x in zip(counts, bin_centers):
            # Only put text above bars with counts more than 0
            if count > 0:
                plt.text(x, count, str(int(count)), ha='center', va='bottom')

        plt.savefig(self.log_dir_base + title + ".svg")
        #plt.show()
        return mean_angle, variance_angle

    def write_log_post_training(self, model, test_dataset, test_images, test_labels):
        '''
        Append information to the log after training regarding evaluation metrics and a prediction vs ground truth plot
        :param model:
        :param test_images:
        :param test_labels:
        :return:
        '''
        results = model.evaluate(test_dataset)
        with open(self.log_dir_base + "info.txt", 'a') as log:
            log.write("\n\nResults For Evaluation with Test Set:\n")
            log.write(f"Test MSE: {results[0]}, Test MAE: {results[1]}, Test RMSE: {results[2]}\n")

        self.create_evaluation_plots(model, test_images, test_labels)

    def create_evaluation_plots(self, model, test_images, test_labels):
        '''
        Plots prediction vs ground truth with test data on trained model
        :param model:
        :param test_images:
        :param test_labels:
        :return:
        '''
        import matplotlib.pyplot as plt
        predictions = model.predict(test_images)
        predictions = predictions.flatten()

        plt.figure(figsize=(10, 10))
        plt.scatter(test_labels, predictions)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.axis('equal')  # This sets the same scale for both axes
        plt.grid(True)
        plt.savefig(self.log_dir_base + "Actual vs Predicted Values.svg")

        # Optionally, plot the residuals
        plt.figure(figsize=(10, 10))
        plt.scatter(test_labels, test_labels - predictions)
        plt.xlabel('Actual Values')
        plt.ylabel('Residuals')
        plt.title('Residuals of Predictions')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.axis('equal')  # This sets the same scale for both axes
        plt.grid(True)  # Optionally, add a grid for easier visualization
        plt.savefig(self.log_dir_base + "Residuals of Predictions")



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



    pilot_net = PilotNet(data_dirs=data_dirs, save_dir="/home/luca/raspicar/training/")
    pilot_net.train()
