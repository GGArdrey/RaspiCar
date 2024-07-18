import zmq
import time
from utils.message_utils import create_json_message, parse_json_message
from Node import Node

class XboxGamepadTranslatorNode(Node):
    def __init__(self, zmq_sub_url="tcp://*:5556", zmq_sub_topic="gamepad", zmq_pub_url="tcp://*:5557"):
        super().__init__()
        self.zmq_context = zmq.Context()
        self.zmq_subscriber = self.zmq_context.socket(zmq.SUB)
        self.zmq_subscriber.connect(zmq_sub_url)
        self.zmq_subscriber.setsockopt_string(zmq.SUBSCRIBE, zmq_sub_topic)

        self.zmq_publisher = self.zmq_context.socket(zmq.PUB)
        self.zmq_publisher.bind(zmq_pub_url)

        # Define key mappings
        self.key_mappings = {
            "start": "reset_emergency_stop",
            "back": "emergency_stop",
            "A": "start_data_recording",
            "Y": "stop_data_recording",
            "X": "sensors_enable",
            "B": "sensors_disable"
        }

    def start(self):
        while True:
            try:
                message = self.zmq_subscriber.recv_string()
                topic, timestamp, payload = parse_json_message(message)
                if topic == "gamepad":
                    steering_commands, function_commands = self.translate_gamepad_input(payload)
                    if steering_commands:
                        self.zmq_publisher.send(create_json_message(steering_commands, "gamepad_steering_commands"))
                    if function_commands:
                        self.zmq_publisher.send(create_json_message(function_commands, "gamepad_function_commands"))
            except zmq.ZMQError as e:
                print(f"ZMQ error: {e}")
            except Exception as e:
                print(f"Error processing message: {e}")

    def release(self):
        self.zmq_publisher.close()
        self.zmq_subscriber.close()
        self.zmq_context.term()

    def translate_gamepad_input(self, payload):
        # Start with default values
        steering_commands = {
            "steer": 0.0,
            "throttle": 0.0,
            "emergency_stop": 0,
            "reset_emergency_stop": 0,
            "sensors_enable": 0,
            "sensors_disable": 0
        }

        function_commands = {
            "start_data_recording": 0,
            "stop_data_recording": 0
        }

        # Handle joystick and trigger inputs
        if "left_stick_x" in payload:
            steering_commands["steer"] = payload["left_stick_x"]
        if "right_trigger" in payload:
            steering_commands["throttle"] += payload["right_trigger"]
        if "left_trigger" in payload:
            steering_commands["throttle"] -= payload["left_trigger"]

        # Handle button presses for other commands
        for button, command in self.key_mappings.items():
            if payload.get(button):
                if command in steering_commands:
                    steering_commands[command] = 1
                else:
                    function_commands[command] = 1

        return steering_commands, function_commands

if __name__ == "__main__":
    translator = XboxGamepadTranslatorNode()
    try:
        translator.start()
    finally:
        translator.release()
