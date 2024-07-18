import time
from src.utils.message_utils import parse_image_message
from util import timer
from src.utils.pilotnet_utils import resize_and_crop_image
import numpy as np
from tflite_runtime.interpreter import Interpreter
import zmq
from message import create_json_message

class LaneDetectionPilotnet():

    def __init__(self, zmq_pub_url="cp://localhost:5560", zmq_pub_topic = "pilotnetc_steering_commands", camera_sub_url="tcp://localhost:5555", camera_sub_topic="frame"):
        self.latest_frame = None
        self.interpreter = Interpreter(model_path="/home/pi/models/cp-0024.tflite")
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
        self.zmq_subscriber.setsockopt_string(zmq.SUBSCRIBE, self.camera_sub_topic)



    def start(self):
        while True:
            try:
                message_parts = self.zmq_subscriber.recv_multipart()
                topic, timestamp, image = parse_image_message(message_parts)
                self.predict(image)
            except zmq.ZMQError as e:
                print(f"ZMQ error: {e}")
            except Exception as e:
                print(f"Error processing frame: {e}")

    def stop(self):
        self.zmq_publisher.close()
        self.zmq_subscriber.close()
        self.zmq_context.term()


    def predict(self, frame):
        frame = resize_and_crop_image(frame)
        frame = np.array(frame, dtype=np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], [frame])

        with timer("Inference Time"):
            self.interpreter.invoke()
            overall_predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        predicted_class = np.argmax(overall_predictions)
        predicted_steering_value = self.classes[predicted_class]
        predicted_steering_prob = overall_predictions[predicted_class]
        left_prob = overall_predictions[predicted_class - 1] if predicted_class > 0 else 0
        right_prob = overall_predictions[predicted_class + 1] if predicted_class < len(overall_predictions) - 1 else 0
        weighted_steering_prob = predicted_steering_prob + left_prob + right_prob
        weighted_steering_value = (left_prob * self.classes[predicted_class - 1] if predicted_class > 0 else 0) + \
                               (predicted_steering_prob * self.classes[predicted_class]) + \
                               (right_prob * self.classes[predicted_class + 1] if predicted_class < len(
                                   overall_predictions) - 1 else 0)

        weighted_steering_value = weighted_steering_value / weighted_steering_prob if weighted_steering_prob > 0 else 0

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
            "overall_predictions": overall_predictions
        }

        message = create_json_message(payload, self.zmq_pub_topic)
        self.zmq_publisher.send_json(message)

if __name__ == "__main__":
    pilotnet = LaneDetectionPilotnet()
    pilotnet.start()

