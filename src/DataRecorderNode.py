import cv2
import os
import zmq
from utils.message_utils import parse_image_message, parse_json_message
import numpy as np
from datetime import datetime
from Node import Node

class DataRecorderNode(Node):
    def __init__(self, camera_sub_url="tcp://localhost:5555",
                 camera_sub_topic="camera",
                 gamepad_sub_url="tcp://localhost:5557",
                 gamepad_function_sub_topic="function_commands",
                 gamepad_steering_sub_topic="steering_commands",
                 save_dir='/home/pi/data/'):
        super().__init__()
        self.camera_sub_url = camera_sub_url
        self.camera_sub_topic = camera_sub_topic
        self.gamepad_sub_url = gamepad_sub_url
        self.gamepad_function_sub_topic = gamepad_function_sub_topic
        self.gamepad_steering_sub_topic = gamepad_steering_sub_topic
        self.save_dir = save_dir
        self.max_time_diff = 0.1  # Maximum allowable time difference between frame and gamepad data
        self.running = False
        self.storage_dir = None
        self.image_count = 0

        # Setup ZeroMQ context and sockets
        self.zmq_context = zmq.Context()

        # Camera subscriber
        self.camera_subscriber = self.zmq_context.socket(zmq.SUB)
        self.camera_subscriber.connect(self.camera_sub_url)
        self.camera_subscriber.setsockopt_string(zmq.SUBSCRIBE, self.camera_sub_topic)

        # Function commands subscriber
        self.function_commands_subscriber = self.zmq_context.socket(zmq.SUB)
        self.function_commands_subscriber.connect(self.gamepad_sub_url)
        self.function_commands_subscriber.setsockopt_string(zmq.SUBSCRIBE, self.gamepad_function_sub_topic)

        # Steering commands subscriber
        self.steering_commands_subscriber = self.zmq_context.socket(zmq.SUB)
        self.steering_commands_subscriber.connect(self.gamepad_sub_url)
        self.steering_commands_subscriber.setsockopt_string(zmq.SUBSCRIBE, self.gamepad_steering_sub_topic)

        # Poller to monitor function commands
        self.poller = zmq.Poller()
        self.poller.register(self.function_commands_subscriber, zmq.POLLIN)

        self.latest_frame = None
        self.latest_frame_timestamp = None
        self.recording = False

    def start(self):
        print("Data Recorder started...")
        try:
            while True:
                events = dict(self.poller.poll())

                if self.function_commands_subscriber in events:
                    self._process_function_commands()

                if self.recording:
                    if self.camera_subscriber in events:
                        self._process_camera_frame()

                    if self.steering_commands_subscriber in events:
                        self._process_steering_commands()

        except KeyboardInterrupt:
            print("Data Recorder interrupted.")
        finally:
            self.release()

    def release(self):
        self.camera_subscriber.close()
        self.function_commands_subscriber.close()
        self.steering_commands_subscriber.close()
        self.zmq_context.term()
        print("Data Recorder released and ZeroMQ subscribers closed.")

    def _process_camera_frame(self):
        message = self.camera_subscriber.recv_multipart()
        topic, frame, timestamp = parse_image_message(message)
        self.latest_frame = frame
        self.latest_frame_timestamp = timestamp

    def _process_function_commands(self):
        message = self.function_commands_subscriber.recv_string()
        topic, timestamp, payload = parse_json_message(message)

        if payload["start_data_recording"] == 1:
            if not self.recording:
                print("Recording started...")
                self.setup_recording_dir()
                self.recording = True
                self.image_count = 0  # Reset counter
                # Register the camera and steering subscribers with the poller
                self.poller.register(self.camera_subscriber, zmq.POLLIN)
                self.poller.register(self.steering_commands_subscriber, zmq.POLLIN)
        elif payload["stop_data_recording"] == 1:
            if self.recording:
                print("Recording stopped...")
                self.recording = False
                # Unregister the camera and steering subscribers from the poller
                self.poller.unregister(self.camera_subscriber)
                self.poller.unregister(self.steering_commands_subscriber)

    def _process_steering_commands(self):
        message = self.steering_commands_subscriber.recv_string()
        topic, timestamp, payload = parse_json_message(message)
        print(timestamp)

        if self.recording and "steer" in payload:
            self._pair_and_save_frame(payload, timestamp)

    def _pair_and_save_frame(self, steering_data, steering_timestamp):
        if self.latest_frame is not None and abs(steering_timestamp - self.latest_frame_timestamp) <= self.max_time_diff:
            self.save_frame_with_label(self.latest_frame, steering_data)
            self.latest_frame = None  # Clear the latest frame after saving

    def setup_recording_dir(self):
        date_time_dir = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.storage_dir = os.path.join(self.save_dir, date_time_dir)
        os.makedirs(self.storage_dir, exist_ok=True)  # Ensure the directory exists


    def save_frame_with_label(self, frame, steering_data):
        # Standardize the steering angle
        steering_angle = steering_data["steer"]

        image_path = os.path.join(self.storage_dir, f"{self.image_count}_{steering_angle}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Saved frame {self.image_count} with steering data {steering_angle}.")
        self.image_count += 1

# Example usage:
if __name__ == "__main__":
    node = DataRecorderNode()
    node.start()
