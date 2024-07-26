import logging

import zmq
import time
from utils.message_utils import create_json_message, parse_json_message
from Node import Node
from CommandInterface import CommandInterface  # Import the CommandInterface

# TODO should this node start even when no controller command is received?

class ControlFusionNode(Node):
    def __init__(self, control_sub_url="tcp://*:5560", control_sub_topic="pilotnetc_steering_commands",
                 gamepad_sub_url="tcp://*:5557", gamepad_sub_topic="gamepad_steering_commands",
                 zmq_pub_url="tcp://*:5570", zmq_pub_topic="fused_steering_commands", override_duration=3,
                 log_level=logging.INFO):
        super().__init__(log_level=log_level)
        self.zmq_context = zmq.Context()

        # Control algorithm subscriber
        self.control_subscriber = self.zmq_context.socket(zmq.SUB)
        self.control_subscriber.connect(control_sub_url)
        self.control_subscriber.setsockopt_string(zmq.SUBSCRIBE, control_sub_topic)
        self.control_subscriber.setsockopt(zmq.RCVHWM, 1)  # Set high water mark to 1 to drop old frames

        # Gamepad subscriber
        self.gamepad_subscriber = self.zmq_context.socket(zmq.SUB)
        self.gamepad_subscriber.connect(gamepad_sub_url)
        self.gamepad_subscriber.setsockopt_string(zmq.SUBSCRIBE, gamepad_sub_topic)
        self.gamepad_subscriber.setsockopt(zmq.RCVHWM, 1)  # Set high water mark to 1 to drop old frames

        # Publisher
        self.zmq_pub_topic = zmq_pub_topic
        self.zmq_publisher = self.zmq_context.socket(zmq.PUB)
        self.zmq_publisher.bind(zmq_pub_url)

        # Poller
        self.poller = zmq.Poller()
        self.poller.register(self.control_subscriber, zmq.POLLIN)
        self.poller.register(self.gamepad_subscriber, zmq.POLLIN)

        # Initialize last received messages and override tracking
        self.override_end_times = {
            "steer": 0,
            "throttle": 0,
            "emergency_stop": 0,
            "reset_emergency_stop": 0,
            "sensors_enable": 0,
            "sensors_disable": 0
        }
        self.override_duration = override_duration
        self.default_values = {
            "steer": 0.0,
            "throttle": 0.0,
            "emergency_stop": 0,
            "reset_emergency_stop": 0,
            "sensors_enable": 0,
            "sensors_disable": 0
        }
        self.current_control_msg = None
        self.current_gamepad_msg = dict(self.default_values)

        # Initialize CommandInterface for UART communication
        self.command_interface = CommandInterface()

    def start(self):
        while True:
            events = dict(self.poller.poll())
            current_time = time.time()

            if self.gamepad_subscriber in events:
                self._process_gamepad_message(current_time)

            if self.control_subscriber in events:
                self._process_control_message(current_time)

            fused_msg = self.fuse_messages(current_time)
            self.execute_commands(fused_msg)
            self.publish_message(fused_msg)

    def _process_gamepad_message(self, current_time):
        message = self.gamepad_subscriber.recv_string()
        _, timestamp, payload = parse_json_message(message)
        self.current_gamepad_msg = payload
        self.update_override_end_times(current_time)
        latency = time.time() - timestamp
        if latency > 0.2:
            self.log(f"High latency detected from processing gamepad: {latency}", logging.WARNING)

    def _process_control_message(self, current_time):
        message = self.control_subscriber.recv_string()
        _, timestamp, payload = parse_json_message(message)
        self.update_override_end_times(current_time)
        self.current_control_msg = payload
        latency = time.time() - timestamp
        self.log(f"Total time to process image: {latency}", logging.DEBUG)
        if latency > 0.2:
            self.log(f"High latency detected from processing image: {latency}", logging.WARNING)

    def update_override_end_times(self, current_time):
        for key in self.override_end_times:
            if self.current_gamepad_msg.get(key) != self.default_values.get(key):
                self.override_end_times[key] = current_time + self.override_duration

    def fuse_messages(self, current_time):
        """Fuse messages from control algorithm and gamepad based on override end times."""
        fused_msg = {}

        for key in self.default_values:
            # Use gamepad value if within override duration
            if self.current_gamepad_msg and current_time <= self.override_end_times[key]:
                fused_msg[key] = self.current_gamepad_msg[key]
            # Otherwise, use control algorithm value
            elif self.current_control_msg:
                fused_msg[key] = self.current_control_msg[key]
            else:
                fused_msg[key] = self.default_values[key]

        return fused_msg

    def execute_commands(self, message):
        """Execute the fused commands via UART."""
        if "steer" in message:
            self.command_interface.steer(message["steer"])
        if "throttle" in message:
            self.command_interface.throttle(message["throttle"])
        if "sensors_enable" in message and message["sensors_enable"]:
            self.command_interface.sensors_enable()
        if "sensors_disable" in message and message["sensors_disable"]:
            self.command_interface.sensors_disable()
        if "emergency_stop" in message and message["emergency_stop"]:
            self.command_interface.emergency_stop()
        if "reset_emergency_stop" in message and message["reset_emergency_stop"]:
            self.command_interface.reset_emergency_stop()  # or handle reset differently if needed

    def publish_message(self, message):
        """Publish the fused message."""
        self.zmq_publisher.send(create_json_message(message, self.zmq_pub_topic))

    def release(self):
        """Clean up and close sockets and context."""
        self.zmq_publisher.close()
        self.control_subscriber.close()
        self.gamepad_subscriber.close()
        self.zmq_context.term()


if __name__ == "__main__":
    fusion_node = ControlFusionNode()
    try:
        fusion_node.start()
    finally:
        fusion_node.release()
