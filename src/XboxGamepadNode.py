import time
import pygame
import zmq
import json
from Node import Node

class XboxGamepadNode(Node):
    def __init__(self, joystick_index=0, input_freq=10, zmq_pub_url="tcp://*:5556"):
        Node.__init__(self)
        pygame.init()
        pygame.joystick.init()
        self._input_freq = input_freq
        self.joystick_index = joystick_index
        self.zmq_pub_url = zmq_pub_url
        self.zmq_context = zmq.Context()
        self.zmq_publisher = self.zmq_context.socket(zmq.PUB)
        self.zmq_publisher.bind(self.zmq_pub_url)
        self.connected = self._connect_gamepad(joystick_index)

    def start(self):
        print("Started Xbox Input Node")
        while True:
            for event in pygame.event.get():  # Get the list of all events
                if event.type == pygame.JOYDEVICEREMOVED:
                    self._handle_disconnect()
                    continue  # Skip further processing after disconnection

            if not self.connected:
                time.sleep(1)  # Wait a bit before trying to reconnect
                self.connected = self._connect_gamepad(self.joystick_index)
                if not self.connected:
                    print("Attempt to reconnect controller failed.")
                    continue  # Skip further processing if not connected

            self._publish_gamepad_input()
            time.sleep(1 / self._input_freq)  # Control the polling rate

    def release(self):
        pygame.quit()
        self.zmq_publisher.close()
        self.zmq_context.term()
        print("Xbox Input Node released and ZeroMQ publisher closed.")

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
                print("Controller connected.")
                return True
            except pygame.error as e:
                print(f"Controller connection error: {e}")
        else:
            print(f"No controller found at index {joystick_index}.")
        return False

    def _handle_disconnect(self):
        """Handle controller disconnection. Sets state of self.connected to false to indicate"""
        print("Controller disconnected.")
        if self._joystick:
            self._joystick.quit()  # Properly close the joystick instance
            self._joystick = None
        self.connected = False

    def _publish_gamepad_input(self):
        """Handle the input from the controller."""
        data = {
            "left_stick_x": self._joystick.get_axis(0),  # Axis 0: left joystick left-right
            "left_stick_y": self._joystick.get_axis(1),  # Axis 1: left joystick up-down
            "right_stick_x": self._joystick.get_axis(3),  # Axis 3: right joystick left-right
            "right_stick_y": self._joystick.get_axis(4),  # Axis 4: right joystick up-down
            "left_trigger": self._joystick.get_axis(2),  # Axis 2: left trigger
            "right_trigger": self._joystick.get_axis(5),  # Axis 5: right trigger

            "A": self._joystick.get_button(0),  # Button 0: A
            "B": self._joystick.get_button(1),  # Button 1: B
            "X": self._joystick.get_button(2),  # Button 2: X
            "Y": self._joystick.get_button(3),  # Button 3: Y
            "left_bumper": self._joystick.get_button(4),  # Button 4: Left Bumper
            "right_bumper": self._joystick.get_button(5),  # Button 5: Right Bumper
            "back": self._joystick.get_button(6),  # Button 6: Back
            "start": self._joystick.get_button(7),  # Button 7: Start
            "left_stick_click": self._joystick.get_button(9),  # Button 9: Left Stick Click
            "right_stick_click": self._joystick.get_button(10),  # Button 10: Right Stick Click
            "dpad_up": self._joystick.get_button(13),  # Button 13: D-Pad Up
            "dpad_down": self._joystick.get_button(14),  # Button 14: D-Pad Down
            "dpad_left": self._joystick.get_button(11),  # Button 11: D-Pad Left
            "dpad_right": self._joystick.get_button(12)  # Button 12: D-Pad Right
        }

        timestamp = time.time()
        message = {
            "timestamp": timestamp,
            "data": data
        }
        message_json = json.dumps(message)
        self.zmq_publisher.send_string(f"controller_state {message_json}")
