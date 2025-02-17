"""
RaspiCar
Copyright (c) 2024 Fynn Luca Maaß

Licensed under the Custom License. See the LICENSE file in the project root for license terms.
"""

import logging
import threading

import cv2
import zmq
import time
from utils.message_utils import create_json_message, parse_json_message, parse_jpg_image_message
from Node import Node
import tensorflow as tf
from source.PilotNetJunctionBase import BasePilotNet, reverse_gradient, GradientReversal, stack_branches_func, mask_outputs, resize_and_crop_image_tf, preprocess_image
import numpy as np

class PilotNetJunctionNode(Node):
    '''
    This is a simple node to subscribe to camera images and display them. Run it on your own computer to display the
    RaspiCars camera.
    '''

    def __init__(self, log_level=logging.INFO,
                 zmq_pub_url="tcp://localhost:5560",
                 zmq_pub_topic="steering_commands",
                 camera_sub_url="tcp://raspberrypi.local:5550",
                 camera_sub_topic="camera",
                 gamepad_sub_url="tcp://raspberrypi.local:5540",
                 gamepad_sub_topic="gamepad_steering_commands",
                 use_weighted_average=False):
        super().__init__(log_level=log_level)
        self.zmq_context = zmq.Context()

        self.camera_subscriber = self.zmq_context.socket(zmq.SUB)
        self.camera_subscriber.connect(camera_sub_url)
        self.camera_subscriber.setsockopt(zmq.RCVHWM, 1)  # Set high water mark to 1 to drop old frames
        self.camera_subscriber.setsockopt(zmq.CONFLATE, 1)  # Keep only the latest message
        self.camera_subscriber.setsockopt_string(zmq.SUBSCRIBE, camera_sub_topic)

        # Gamepad subscriber
        self.gamepad_subscriber = self.zmq_context.socket(zmq.SUB)
        self.gamepad_subscriber.connect(gamepad_sub_url)
        self.gamepad_subscriber.setsockopt(zmq.RCVHWM, 1)  # Set high water mark to 1 to drop old frames
        self.gamepad_subscriber.setsockopt(zmq.CONFLATE, 1)  # Keep only the latest message
        self.gamepad_subscriber.setsockopt_string(zmq.SUBSCRIBE, gamepad_sub_topic)

        self.zmq_publisher = self.zmq_context.socket(zmq.PUB)
        self.zmq_publisher.bind(zmq_pub_url)
        self.zmq_pub_topic = zmq_pub_topic

        # Poller
        self.poller = zmq.Poller()
        self.poller.register(self.camera_subscriber, zmq.POLLIN)
        self.poller.register(self.gamepad_subscriber, zmq.POLLIN)

        self.default_values = {
            "steer": 0.0,
            "control_command": "LANEFOLLOW",
            "throttle": 0.0,
            "emergency_stop": 0,
            "reset_emergency_stop": 0,
            "sensors_enable": 0,
            "sensors_disable": 0
        }
        self.current_gamepad_msg = dict(self.default_values)

        self.PilotNet = BasePilotNet(params={
            "data_dirs": {"": 0},
            "save_dir": "",
            "target_width": 200,
            "target_height": 66,
            "batch_size": 128,
            "epochs": 100,
            "initial_learning_rate": 1e-3,
            "domain_loss_weight": 0.0
            },
            create_directories=False)

        self.use_weighted_average = use_weighted_average

    def start(self):
        while True:
            try:
                events = dict(self.poller.poll())
                current_time = time.time()

                if self.gamepad_subscriber in events:
                    self._process_gamepad_message()
                if self.camera_subscriber in events:
                    self._process_camera_message()

            except Exception as e:
                self.log(f"Error receiving frame: {e}", logging.ERROR)
                break

    def _process_gamepad_message(self):
        message = self.gamepad_subscriber.recv_string()
        _, timestamp, payload = parse_json_message(message)
        self.current_gamepad_msg = payload

    def _process_camera_message(self):
        # Receive and Render Image
        message = self.camera_subscriber.recv_multipart()
        topic, image, timestamp = parse_jpg_image_message(message)
        if image is not None:
            cv2.imshow('Received Camera Feed', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

        # Inference on model with image
        image = preprocess_image(image, self.PilotNet.target_height, self.PilotNet.target_width)
        image_tensor = preprocess_image(
            image,
            self.PilotNet.target_width,
            self.PilotNet.target_height
        )
        image_tensor = tf.expand_dims(image_tensor, axis=0)
        road_option = self.current_gamepad_msg.get("control_command", "LANEFOLLOW")
        command_tensor = self._preprocess_command(road_option)
        predictions = self.model.predict(
            {'image_input': image_tensor, 'command_input': command_tensor},
            verbose=0,
        )

        # Calculate steering angle using max likelihood or weighted average
        class_probs = predictions["class_output"]
        if self.use_weighted_average:
            pilotnet_steering_angle = np.dot(class_probs[0], self.PilotNet.boundaries)
        else:
            steering_class = np.argmax(class_probs[0])
            pilotnet_steering_angle = self.PilotNet.boundaries[steering_class]

        # Publish steering commands
        payload = {
            "steer": pilotnet_steering_angle,
            "control_command": road_option,
            "throttle": 0,
            "emergency_stop": 0,
            "reset_emergency_stop": 0,
            "sensors_enable": 0,
            "sensors_disable": 0,
            "overall_predictions": class_probs.tolist()  # Convert numpy array to list for JSON serialization
        }
        self.log(f"Steering Prediction: {pilotnet_steering_angle}", logging.DEBUG)
        message = create_json_message(payload, self.zmq_pub_topic, timestamp=timestamp)
        self.zmq_publisher.send(message)

    def _preprocess_command(self, road_option):
        """
        Turns the road_option enum into one-hot for the model.
        """
        cmd_idx = self.PilotNet.control_command_to_index.get(
            road_option,
            self.PilotNet.control_command_to_index['LANEFOLLOW']
        )
        cmd_one_hot = tf.keras.utils.to_categorical(
            cmd_idx,
            num_classes=self.PilotNet.total_control_commands
        )
        return tf.convert_to_tensor(np.expand_dims(cmd_one_hot, axis=0), dtype=tf.float32)


    def _load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        # Get the input shape of the image input layer
        image_input_shape = self.model.input_shape[1]
        # Extract height and width
        target_height, target_width = image_input_shape[1], image_input_shape[2]
        print(f"Expected input size: {target_height}x{target_width}")
        print("Model layers:")
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                print(f"- {layer.name}")


    def release(self):
        self.camera_subscriber.close()
        self.gamepad_subscriber.close()
        self.zmq_context.term()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    subscriber_node = PilotNetJunctionNode(log_level=logging.DEBUG, use_weighted_average=False)
    try:
        subscriber_node.start()
    except KeyboardInterrupt:
        pass
    finally:
        subscriber_node.release()
