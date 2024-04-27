import time
import pygame
from CarCommands import CarCommands
from IInputSource import IInputSource
from threading import Thread, Lock

'''
XBOX BUTTON MAPPINGS:
A -> button 0
B -> button 1
X -> button 2
Y -> button 3

Right Joystick:
left-right -> axis 3 (neutral 0; left -1; right 1)
up-down -> axis 4 (neutral 0; up -1; down 1)
click -> button 10

Left Joystick:
left-right -> axis 0 (neutral 0; left -1; right 1)
up-down -> axis 1 (neutral 0; up -1; down 1)
click -> button 9

start -> button 7
back -> button 6

D-Pad:
up -> button 13
down -> button 12
left -> button 14
right -> button 11

Right Shoulder
top -> button 5
bottom -> axis 5 (default -1; fully pressed 1)

Left Shoulder
top -> button 4
bottom -> axis 2 (default -1; fully pressed 1)
'''

class XboxInput(IInputSource):

    def __init__(self, joystick_index=0, input_freq=10):
        pygame.init()
        pygame.joystick.init()
        self._input_freq = input_freq
        self.joystick_index = joystick_index
        self.car_commands = CarCommands()
        self.lock = Lock()
        self.running = True  # Flag to control the thread loop
        self.connected = self._connect_controller(joystick_index)
        self.thread = Thread(target=self._poll_inputs)
        self.thread.start()

    def _deinit(self):
        self.running = False
        self.thread.join()  # Ensure the thread has finished before quitting pygame
        pygame.quit()

    def _rumble_controller(self):
        """Rumble the connected controller to provide feedback."""
        if self.connected:
            self._joystick.rumble(1.0, 1.0, 200)  # may need adjusting
            time.sleep(0.2)
            self._joystick.rumble(0, 0, 0)  # Stop rumbling

    def _connect_controller(self,joystick_index=0):
        """Connect to the specified joystick/controller.
        :returns True if connected successfully"""
        if pygame.joystick.get_count() > 0:
            try:
                self._joystick = pygame.joystick.Joystick(joystick_index)
                self._joystick.init()
                self.connected = True
                self._rumble_controller()
                print("Controller connected.")
                return True
            except pygame.error as e:
                print(f"Controller connection error: {e}")
        else:
            print(f"No controller found at index {joystick_index}.")
        return False

    def _poll_inputs(self):
        """Poll inputs from the controller in a separate thread."""
        while self.running:
            for event in pygame.event.get():  # Get the list of all events
                if event.type == pygame.JOYDEVICEREMOVED:
                    self._handle_disconnect()
                    continue  # Skip further processing after disconnection

            if not self.connected:
                time.sleep(1)  # Wait a bit before trying to reconnect
                self.connected = self._connect_controller(self.joystick_index)
                if not self.connected:
                    print("Attempt to reconnect controller failed.")
                    continue  # Skip further processing if not connected

            self._handle_controller_input()

            time.sleep(1 / self._input_freq)  # Control the polling rate

    def _handle_disconnect(self):
        """Handle controller disconnection. Sets state of self.connected to false to indicate"""
        with self.lock:
            print("Controller disconnected.")
            self.car_commands.stop = True  # Signal to stop the vehicle
            if self._joystick:
                self._joystick.quit()  # Properly close the joystick instance
                self._joystick = None
            self.connected = False

    def _handle_controller_input(self):
        """Handle the input from the controller."""
        with self.lock:
            car_commands = CarCommands()
            # Update throttle based on triggers
            forward_throttle = (self._joystick.get_axis(5) + 1) / 2  # Axis 5: RT, range [-1, 1], remapped to [0, 1]
            backward_throttle = -((self._joystick.get_axis(2) + 1) / 2)  # Axis 2: LT, range [-1, 1], remapped to [-1, 0]
            car_commands.throttle = forward_throttle + backward_throttle  # Combine forward and backward throttle

            # Update steering based on the left joystick horizontal axis
            car_commands.steer = self._joystick.get_axis(0)  # Axis 0: left joystick left-right

            # Check for "Y" button press to activate a stop command
            car_commands.stop = self._joystick.get_button(3)  # Button 3: Y button

            self.car_commands = car_commands


    def read_inputs(self) -> CarCommands:
        """Read inputs from the controller."""
        with self.lock:
            return self.car_commands.copy()  # Assuming CarCommands has a `copy` method

