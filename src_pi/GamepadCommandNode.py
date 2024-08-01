import zmq
import time
from utils.message_utils import create_json_message, parse_json_message
from Node import Node
import logging

class GamepadCommandNode(Node):
    def __init__(self, log_level=logging.INFO,
                 joystick_deadzone = 0.15,
                 gamepad_sub_url="tcp://localhost:5540",
                 gamepad_sub_topic="gamepad",
                 gamepad_command_pub_url="tcp://localhost:5541",
                 gamepad_function_commands_pub_topic = "gamepad_function_commands",
                 gamepad_steering_commands_pub_topic = "gamepad_steering_commands"):
        super().__init__(log_level=log_level)
        self.joystick_deadzone = joystick_deadzone
        self.gamepad_function_commands_pub_topic = gamepad_function_commands_pub_topic
        self.gamepad_steering_commands_pub_topic = gamepad_steering_commands_pub_topic
        self.zmq_context = zmq.Context()
        self.gamepad_subscriber = self.zmq_context.socket(zmq.SUB)
        self.gamepad_subscriber.connect(gamepad_sub_url)
        self.gamepad_sub_topic = gamepad_sub_topic
        self.gamepad_sub_topic_disconnected = gamepad_sub_topic + "_disconnected"
        self.gamepad_subscriber.setsockopt_string(zmq.SUBSCRIBE, self.gamepad_sub_topic)
        self.gamepad_subscriber.setsockopt_string(zmq.SUBSCRIBE, self.gamepad_sub_topic_disconnected)

        self.zmq_publisher = self.zmq_context.socket(zmq.PUB)
        self.zmq_publisher.bind(gamepad_command_pub_url)

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
                message = self.gamepad_subscriber.recv_string()
                topic, timestamp, payload = parse_json_message(message)
                if topic == self.gamepad_sub_topic:
                    steering_commands, function_commands = self.translate_gamepad_input(payload)
                    if steering_commands:
                        self.zmq_publisher.send(create_json_message(steering_commands, self.gamepad_steering_commands_pub_topic, timestamp))
                    if function_commands:
                        self.zmq_publisher.send(create_json_message(function_commands, self.gamepad_function_commands_pub_topic, timestamp))
                elif topic == self.gamepad_sub_topic_disconnected:
                    steering_commands = {
                        "steer": 0.0,
                        "throttle": 0.0,
                        "emergency_stop": 1,
                        "reset_emergency_stop": 0,
                        "sensors_enable": 0,
                        "sensors_disable": 0
                    }
                    self.zmq_publisher.send(create_json_message(steering_commands, "gamepad_steering_commands", timestamp))
            except zmq.ZMQError as e:
                self.log(f"ZMQ error: {e}", logging.ERROR)
            except Exception as e:
                self.log(f"{e}", logging.ERROR)

    def release(self):
        self.zmq_publisher.close()
        self.gamepad_subscriber.close()
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
            steer_value = payload["left_stick_x"]
            steering_commands["steer"] = 0 if abs(steer_value) <= self.joystick_deadzone else steer_value
        if "right_trigger" in payload:
            steering_commands["throttle"] += (payload["right_trigger"] + 1) / 2 # mapping raw inputs
        if "left_trigger" in payload:
            steering_commands["throttle"] -= (payload["left_trigger"] + 1) / 2 # mapping raw inputs

        # Handle button presses for other commands
        for button, command in self.key_mappings.items():
            if payload.get(button):
                if command in steering_commands:
                    steering_commands[command] = 1
                else:
                    function_commands[command] = 1

        return steering_commands, function_commands

if __name__ == "__main__":
    translator = GamepadCommandNode()
    try:
        translator.start()
    finally:
        translator.release()
