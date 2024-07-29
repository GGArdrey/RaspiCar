import time
import pygame
import zmq
from utils.message_utils import create_json_message
from Node import Node
import logging
pygame.mixer.pre_init(frequency=48000, buffer=2048)


class XboxGamepadNode(Node):
    def __init__(self,log_level=logging.INFO,
                 joystick_index=0,
                 input_freq=20,
                 zmq_pub_url="tcp://localhost:5540",
                 pub_topic="gamepad"):
        super().__init__(log_level=log_level)
        pygame.init()
        pygame.joystick.init()
        self._input_freq = input_freq
        self.joystick_index = joystick_index
        self.zmq_pub_url = zmq_pub_url
        self.pub_topic = pub_topic
        self.pub_topic_disconnected = pub_topic + "_disconnected"
        self.zmq_context = zmq.Context()
        self.zmq_publisher = self.zmq_context.socket(zmq.PUB)
        self.zmq_publisher.bind(self.zmq_pub_url)
        self.connected = self._connect_gamepad(joystick_index)

        # Initialize previous state
        self.previous_state = self._get_initial_state()

    def start(self):
        while True:
            for event in pygame.event.get():  # Get the list of all events
                if event.type == pygame.JOYDEVICEREMOVED:
                    self._handle_disconnect()
                    continue  # Skip further processing after disconnection

            if not self.connected:
                self._publish_gamepad_disconnect()
                time.sleep(1)  # Wait a bit before trying to reconnect
                self.connected = self._connect_gamepad(self.joystick_index)
                if not self.connected:
                    self.log("Attempt to reconnect controller failed.", logging.ERROR)
                    continue  # Skip further processing if not connected

            self._publish_gamepad_input()
            time.sleep(1 / self._input_freq)  # Control the polling rate

    def release(self):
        pygame.quit()
        self.zmq_publisher.close()
        self.zmq_context.term()

    def _rumble_gamepad(self):
        """Rumble the connected controller to provide feedback."""
        if self.connected:
            self._joystick.rumble(1.0, 1.0, 200)  # may need adjusting
            time.sleep(0.2)
            self._joystick.rumble(0, 0, 0)  # Stop rumbling

    def _connect_gamepad(self, joystick_index=0):
        """Connect to the specified joystick/controller.
        :returns True if connected successfully"""
        if pygame.joystick.get_count() > 0:
            try:
                self._joystick = pygame.joystick.Joystick(joystick_index)
                self._joystick.init()
                self.connected = True
                self._rumble_gamepad()
                self.log("Controller connected.", logging.INFO)
                return True
            except pygame.error as e:
                self.log(f"Controller connection error: {e}", logging.ERROR)
        else:
            self.log(f"No controller found at index {joystick_index}.", logging.ERROR)
        return False

    def _handle_disconnect(self):
        """Handle controller disconnection. Sets state of self.connected to false to indicate"""
        if self._joystick:
            self._joystick.quit()  # Properly close the joystick instance
            self._joystick = None
        self.connected = False

    def _get_initial_state(self):
        """Get the initial state of the gamepad."""
        return {
            "left_stick_x": 0.0,
            "left_stick_y": 0.0,
            "right_stick_x": 0.0,
            "right_stick_y": 0.0,
            "left_trigger": 0.0,
            "right_trigger": 0.0,
            "A": 0,
            "B": 0,
            "X": 0,
            "Y": 0,
            "left_bumper": 0,
            "right_bumper": 0,
            "back": 0,
            "start": 0,
            "left_stick_click": 0,
            "right_stick_click": 0,
            "dpad_up": 0,
            "dpad_down": 0,
            "dpad_left": 0,
            "dpad_right": 0
        }

    def _publish_gamepad_disconnect(self):
        """just publish the default message on discoonnect topic"""
        default_state = self._get_initial_state()
        message = create_json_message(default_state, self.pub_topic_disconnected)
        self.zmq_publisher.send(message)

    def _publish_gamepad_input(self):
        """Handle the input from the controller."""
        current_state = {
            "left_stick_x": self._joystick.get_axis(0),
            "left_stick_y": self._joystick.get_axis(1),
            "right_stick_x": self._joystick.get_axis(3),
            "right_stick_y": self._joystick.get_axis(4),
            "left_trigger": self._joystick.get_axis(2),
            "right_trigger": self._joystick.get_axis(5),
            "A": self._joystick.get_button(0),
            "B": self._joystick.get_button(1),
            "X": self._joystick.get_button(2),
            "Y": self._joystick.get_button(3),
            "left_bumper": self._joystick.get_button(4),
            "right_bumper": self._joystick.get_button(5),
            "back": self._joystick.get_button(6),
            "start": self._joystick.get_button(7),
            "left_stick_click": self._joystick.get_button(9),
            "right_stick_click": self._joystick.get_button(10),
            "dpad_up": self._joystick.get_button(13),
            "dpad_down": self._joystick.get_button(14),
            "dpad_left": self._joystick.get_button(11),
            "dpad_right": self._joystick.get_button(12)
        }

        #if current_state != self.previous_state:
        message = create_json_message(current_state, self.pub_topic)
        self.zmq_publisher.send(message)
        self.previous_state = current_state
