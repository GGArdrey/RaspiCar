import cv2
import os
import zmq
import json
import numpy as np
from datetime import datetime
from Node import Node

class DataRecorderNode(Node):
    def __init__(self, camera_sub_url="tcp://localhost:5555", controller_sub_url="tcp://localhost:5556",
                 save_dir='/home/pi/data/'):
        super().__init__()
        self.camera_sub_url = camera_sub_url
        self.controller_sub_url = controller_sub_url
        self.save_dir = save_dir
        self.max_time_diff = 0.1  # Maximum allowable time difference between frame and joystick data
        self.running = False
        self.storage_dir = None
        self.image_count = 0

        # Setup ZeroMQ context and sockets
        self.zmq_context = zmq.Context()

        # Camera subscriber
        self.camera_subscriber = self.zmq_context.socket(zmq.SUB)
        self.camera_subscriber.connect(self.camera_sub_url)
        self.camera_subscriber.setsockopt_string(zmq.SUBSCRIBE, 'frame')

        # Controller subscriber
        self.controller_subscriber = self.zmq_context.socket(zmq.SUB)
        self.controller_subscriber.connect(self.controller_sub_url)
        self.controller_subscriber.setsockopt_string(zmq.SUBSCRIBE, 'controller_state')

        # Poller to monitor both sockets
        self.poller = zmq.Poller()
        self.poller.register(self.camera_subscriber, zmq.POLLIN)
        self.poller.register(self.controller_subscriber, zmq.POLLIN)

        self.latest_frame = None
        self.latest_frame_timestamp = None
        self.latest_joystick_data = None
        self.recording = False

    def start(self):
        print("Data Recorder started...")
        try:
            while True:
                events = dict(self.poller.poll())

                if self.controller_subscriber in events:
                    self._process_controller_input()

                if self.camera_subscriber in events:
                    self._process_camera_frame()

        except KeyboardInterrupt:
            print("Data Recorder interrupted.")
        finally:
            self.release()

    def release(self):
        self.camera_subscriber.close()
        self.controller_subscriber.close()
        self.zmq_context.term()
        print("Data Recorder released and ZeroMQ subscribers closed.")

    def _process_camera_frame(self):
        topic, frame, timestamp = self.camera_subscriber.recv_multipart()
        timestamp = float(timestamp.decode('utf-8'))
        frame = np.frombuffer(frame, dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        self.latest_frame = frame
        self.latest_frame_timestamp = timestamp

    def _process_controller_input(self):
        message = self.controller_subscriber.recv_string()
        topic, payload = message.split(' ', 1)  # Split the topic and the JSON payload
        controller_data = json.loads(payload)
        timestamp = controller_data["timestamp"]
        data = controller_data["data"]

        if data["dpad_up"] == 1:
            if not self.recording:
                print("Recording started...")
                self.start_recording()
                self.recording = True
        elif data["dpad_down"] == 1:
            if self.recording:
                print("Recording stopped...")
                self.recording = False

        if self.recording:
            self._pair_and_save_frame(data, timestamp)

    def _pair_and_save_frame(self, joystick_data, joystick_timestamp):
        if self.latest_frame is not None and abs(joystick_timestamp - self.latest_frame_timestamp) <= self.max_time_diff:
            self.save_frame_with_label(self.latest_frame, joystick_data)
            self.latest_frame = None  # Clear the latest frame after saving

    def start_recording(self):
        date_time_dir = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.storage_dir = os.path.join(self.save_dir, date_time_dir)
        os.makedirs(self.storage_dir, exist_ok=True)  # Ensure the directory exists
        self.image_count = 0  # Reset counter

    def save_frame_with_label(self, frame, joystick_data):
        # Standardize the steering angle
        steering_angle = joystick_data["left_stick_x"]

        image_path = os.path.join(self.storage_dir, f"{self.image_count}_{steering_angle}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Saved frame {self.image_count} with joystick data.")
        self.image_count += 1

# Example usage:
if __name__ == "__main__":
    node = DataRecorderNode()
    node.start()
