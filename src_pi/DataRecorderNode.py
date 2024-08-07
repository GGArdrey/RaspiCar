"""
RaspiCar
Copyright (c) 2024 Fynn Luca Maaß

Licensed under the Custom License. See the LICENSE file in the project root for license terms.
"""

import cv2
import os
import zmq
from utils.message_utils import parse_jpg_image_message, parse_json_message
import logging
from datetime import datetime
from Node import Node

class DataRecorderNode(Node):
    def __init__(self, log_level=logging.INFO,
                 max_time_diff=0.1,
                 save_dir='/home/pi/data/',
                 image_sub_url="tcp://localhost:5550",
                 image_sub_topic="camera",
                 gamepad_function_sub_url="tcp://localhost:5541",
                 gamepad_function_sub_topic="gamepad_function_commands",
                 steering_commands_url="tcp://localhost:5541",
                 steering_commands_topic="gamepad_steering_commands"):

        super().__init__(log_level=log_level)
        self.image_sub_url = image_sub_url
        self.image_sub_topic = image_sub_topic
        self.gamepad_function_sub_url = gamepad_function_sub_url
        self.gamepad_function_sub_topic = gamepad_function_sub_topic
        self.steering_commands_sub_url = steering_commands_url
        self.steering_commands_sub_topic = steering_commands_topic
        self.save_dir = save_dir
        self.max_time_diff = max_time_diff  # Maximum allowable time difference between frame and gamepad data
        self.running = False
        self.storage_dir = None
        self.image_count = 0

        # Setup ZeroMQ context
        self.zmq_context = zmq.Context()

        # Function commands subscriber
        self.function_commands_subscriber = self.zmq_context.socket(zmq.SUB)
        self.function_commands_subscriber.connect(self.gamepad_function_sub_url)
        self.function_commands_subscriber.setsockopt_string(zmq.SUBSCRIBE, self.gamepad_function_sub_topic)

        # Poller to monitor function commands
        self.poller = zmq.Poller()
        self.poller.register(self.function_commands_subscriber, zmq.POLLIN)

        self.latest_frame = None
        self.latest_frame_timestamp = None
        self.latest_steering_data = None
        self.latest_steering_timestamp = None
        self.recording = False

    def start(self):
        while True:
            events = dict(self.poller.poll())

            if self.function_commands_subscriber in events:
                self._process_function_commands()

            if self.recording:
                if self.image_subscriber in events:
                    self._process_camera_frame()

                if self.steering_commands_subscriber in events:
                    self._process_steering_commands()

                # Always try to save the latest frame with the latest steering data
                self._save_frame_if_ready()

    def release(self):
        self.function_commands_subscriber.close()
        if hasattr(self, 'image_subscriber') and self.image_subscriber:
            self.image_subscriber.close()
        if hasattr(self, 'steering_commands_subscriber') and self.steering_commands_subscriber:
            self.steering_commands_subscriber.close()
        self.zmq_context.term()

    def _process_camera_frame(self):
        '''
        Image messages are handled here
        '''
        message = self.image_subscriber.recv_multipart()
        topic, frame, timestamp = parse_jpg_image_message(message)
        self.latest_frame = frame
        self.latest_frame_timestamp = timestamp

    def _process_function_commands(self):
        '''
        Function commands are handled here to start or stop data recording.
        '''
        message = self.function_commands_subscriber.recv_string()
        topic, timestamp, payload = parse_json_message(message)

        if payload.get("start_data_recording") == 1:
            if not self.recording:
                self.log("Recording started...", logging.INFO)
                self.setup_recording_dir()
                self._setup_subscribers()
                self.recording = True
                self.image_count = 0  # Reset counter
        elif payload.get("stop_data_recording") == 1:
            if self.recording:
                self.log("Recording stopped...", logging.INFO)
                self._close_subscribers()
                self.recording = False

    def _process_steering_commands(self):
        '''
        steering command messages are handled here
        '''
        message = self.steering_commands_subscriber.recv_string()
        topic, timestamp, payload = parse_json_message(message)
        self.latest_steering_data = payload
        self.latest_steering_timestamp = timestamp

    def _setup_subscribers(self):
        '''
        If recording is started, sets up subscribers to receive image and steering data
        '''
        # Camera subscriber
        self.image_subscriber = self.zmq_context.socket(zmq.SUB)
        self.image_subscriber.connect(self.image_sub_url)
        self.image_subscriber.setsockopt(zmq.RCVHWM, 1)  # Set high water mark to 1 to drop old frames
        self.image_subscriber.setsockopt(zmq.CONFLATE, 1)  # Keep only the latest message
        self.image_subscriber.setsockopt_string(zmq.SUBSCRIBE, self.image_sub_topic)


        # Steering commands subscriber
        self.steering_commands_subscriber = self.zmq_context.socket(zmq.SUB)
        self.steering_commands_subscriber.connect(self.steering_commands_sub_url)
        self.steering_commands_subscriber.setsockopt(zmq.RCVHWM, 1)  # Set high water mark to 1 to drop old frames
        self.image_subscriber.setsockopt(zmq.CONFLATE, 1)  # Keep only the latest message
        self.steering_commands_subscriber.setsockopt_string(zmq.SUBSCRIBE, self.steering_commands_sub_topic)

        self.poller.register(self.image_subscriber, zmq.POLLIN)
        self.poller.register(self.steering_commands_subscriber, zmq.POLLIN)

    def _close_subscribers(self):
        '''
        Will deinitialize subscribers when recording is stopped
        '''
        if hasattr(self, 'camera_subscriber') and self.image_subscriber:
            self.poller.unregister(self.image_subscriber)
            self.image_subscriber.close()
            self.image_subscriber = None

        if hasattr(self, 'steering_commands_subscriber') and self.steering_commands_subscriber:
            self.poller.unregister(self.steering_commands_subscriber)
            self.steering_commands_subscriber.close()
            self.steering_commands_subscriber = None


    def _save_frame_if_ready(self):
        '''
        checks if steering data and camera frame are timely alligned and svaes them.
        '''
        if self.latest_frame is not None and self.latest_steering_data is not None:
            dt = self.latest_steering_timestamp - self.latest_frame_timestamp
            if abs(dt) <= self.max_time_diff:
                self.save_frame_with_label(self.latest_frame, self.latest_steering_data)
                self.latest_frame = None  # Clear the latest frame after saving
            else:
                self.log(f"Time difference between frame and steering data is too large: {dt}, skipping frame...", logging.WARNING)

    def setup_recording_dir(self):
        '''
        Creates a directory to stor the images
        '''
        date_time_dir = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.storage_dir = os.path.join(self.save_dir, date_time_dir)
        os.makedirs(self.storage_dir, exist_ok=True)  # Ensure the directory exists

    def save_frame_with_label(self, frame, steering_data):
        '''
        Saves the images to disk and sets the filename
        '''
        # Standardize the steering angle
        steering_angle = steering_data["steer"]

        image_path = os.path.join(self.storage_dir, f"{self.image_count}_{steering_angle}.jpg")
        cv2.imwrite(image_path, frame)
        self.log(f"Saved frame {self.image_count} with steering data {steering_angle}.", logging.INFO)
        self.image_count += 1

# Example usage:
if __name__ == "__main__":
    node = DataRecorderNode()
    node.start()
