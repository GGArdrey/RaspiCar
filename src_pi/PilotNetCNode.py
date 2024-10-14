import threading
from utils.timer_utils import timer
from utils.pilotnet_utils import resize_and_crop_image
import numpy as np
import time
from tflite_runtime.interpreter import Interpreter, load_delegate
import zmq
from utils.message_utils import create_json_message, parse_jpg_image_message
import json
import logging
from Node import Node

class PilotNetCNode(Node):

    def __init__(self, log_level=logging.INFO,
                 zmq_pub_url="tcp://localhost:5560",
                 zmq_pub_topic="steering_commands",
                 camera_sub_url="tcp://localhost:5550",
                 camera_sub_topic="camera"):
        super().__init__(log_level=log_level)
        self.latest_frame = None
        self.latest_timestamp = None
        self.lock = threading.Lock()

        # Load the Edge TPU delegate
        self.interpreter = Interpreter(
            model_path="/home/pi/models/cp-0024.tflite",
            experimental_delegates=[load_delegate('libedgetpu.so.1')]
        )

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.classes = [-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]

        # ZeroMQ setup
        self.zmq_pub_url = zmq_pub_url
        self.zmq_pub_topic = zmq_pub_topic
        self.camera_sub_url = camera_sub_url
        self.camera_sub_topic = camera_sub_topic

        self.zmq_context = zmq.Context()
        self.zmq_publisher = self.zmq_context.socket(zmq.PUB)
        self.zmq_publisher.bind(self.zmq_pub_url)

        self.zmq_subscriber = self.zmq_context.socket(zmq.SUB)
        self.zmq_subscriber.connect(self.camera_sub_url)
        self.zmq_subscriber.setsockopt(zmq.RCVHWM, 1)  # Set high water mark to 1 to drop old frames
        self.zmq_subscriber.setsockopt_string(zmq.SUBSCRIBE, self.camera_sub_topic)

    def receive_frames(self):
        while True:
            try:
                message = self.zmq_subscriber.recv_multipart()
                topic, image, timestamp = parse_jpg_image_message(message)
                with self.lock:
                    if self.latest_frame is not None:
                        self.log(f"Dropped Camera Frame", logging.WARNING)
                    self.latest_frame = image
                    self.latest_timestamp = timestamp
            except zmq.ZMQError as e:
                print(f"ZMQ error: {e}")
            except Exception as e:
                print(f"Error receiving frame: {e}")

    def start(self):
        receiver_thread = threading.Thread(target=self.receive_frames)
        receiver_thread.daemon = True  # Daemonize thread
        receiver_thread.start()

        while True:
            frame = None
            timestamp = None
            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame
                    timestamp = self.latest_timestamp
                    self.latest_frame = None  # Clear the latest frame
            if frame is not None:
                self.predict(frame, timestamp)
            else:
                time.sleep(0.01)  # Sleep briefly to avoid busy-waiting

    def release(self):
        self.zmq_publisher.close()
        self.zmq_subscriber.close()
        self.zmq_context.term()

    def predict(self, frame, timestamp):
        frame = resize_and_crop_image(frame)
        frame = np.array(frame, dtype=np.float32)

        with timer("Inference Time") as get_elapsed_time:
            self.interpreter.set_tensor(self.input_details[0]['index'], [frame])
            self.interpreter.invoke()
            overall_predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        elapsed_time = get_elapsed_time()
        self.log(f"Inference Time: {elapsed_time}", logging.DEBUG)

        predicted_class = np.argmax(overall_predictions)
        predicted_steering_value = self.classes[predicted_class]
        predicted_steering_prob = overall_predictions[predicted_class]
        left_prob = overall_predictions[predicted_class - 1] if predicted_class > 0 else 0
        right_prob = overall_predictions[predicted_class + 1] if predicted_class < len(overall_predictions) - 1 else 0
        weighted_steering_prob = predicted_steering_prob + left_prob + right_prob
        weighted_steering_value = (
            (left_prob * self.classes[predicted_class - 1] if predicted_class > 0 else 0) +
            (predicted_steering_prob * self.classes[predicted_class]) +
            (right_prob * self.classes[predicted_class + 1] if predicted_class < len(overall_predictions) - 1 else 0)
        )

        weighted_steering_value = weighted_steering_value / weighted_steering_prob if weighted_steering_prob > 0 else 0

        # Steering commands and additional info
        payload = {
            "steer": weighted_steering_value,
            "throttle": 0,
            "emergency_stop": 0,
            "reset_emergency_stop": 0,
            "sensors_enable": 0,
            "sensors_disable": 0,
            "predicted_steering_value": predicted_steering_value,
            "weighted_steering_value": weighted_steering_value,
            "predicted_steering_prob": predicted_steering_prob,
            "weighted_steering_prob": weighted_steering_prob,
            "overall_predictions": overall_predictions.tolist()  # Convert numpy array to list for JSON serialization
        }

        self.log(f"Steering Prediction: {weighted_steering_value}", logging.DEBUG)

        message = create_json_message(payload, self.zmq_pub_topic, timestamp=timestamp)
        self.zmq_publisher.send(message)

if __name__ == "__main__":
    pilotnet = PilotNetCNode()
    pilotnet.start()
