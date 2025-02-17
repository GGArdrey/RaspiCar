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


class CameraSubscriberNode(Node):
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
                 gamepad_sub_topic="gamepad_steering_commands",):
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
        #self.mutex = threading.Lock()

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
        message = self.camera_subscriber.recv_multipart()
        topic, image, timestamp = parse_jpg_image_message(message)
        if image is not None:
            cv2.imshow('Received Camera Feed', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

        # Inference on model with image

        # Publish steering commands

    def _load_model(self, model_path):
        model = tf.keras.models.load_model(model_path, compile=False)
        # Get the input shape of the image input layer
        image_input_shape = model.input_shape[1]
        # Extract height and width
        target_height, target_width = image_input_shape[1], image_input_shape[2]
        print(f"Expected input size: {target_height}x{target_width}")
        print("Model layers:")
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                print(f"- {layer.name}")

        return model

    def release(self):
        self.zmq_subscriber.close()
        self.zmq_context.term()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    subscriber_node = CameraSubscriberNode(log_level=logging.DEBUG, zmq_sub_url="tcp://raspberrypi.local:5550", zmq_sub_topic="camera")
    try:
        subscriber_node.start()
    except KeyboardInterrupt:
        pass
    finally:
        subscriber_node.release()
