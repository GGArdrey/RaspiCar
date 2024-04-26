import time
import pygame
from CarCommands import CarCommands
from IInputSource import IInputSource

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
        self._input_freq = input_freq
        self._clock = pygame.time.Clock()
        self._joystick = None
        self._connect_controller(joystick_index)
        if not self._joystick:
            raise Exception("No controller found.") # TODO handle this case?

    def _deinit(self):
        pygame.quit()

    def _rumble_controller(self):
        """
        Rumble the connected controller to provide feedback.
        """
        self._joystick.rumble(low_frequency=1.0, high_frequency=1.0, duration=200)
        time.sleep(0.2)
        self._joystick.rumble(low_frequency=1.0, high_frequency=1.0, duration=200)

    def _connect_controller(self,joystick_index=0):
        """
        Connect to the specified joystick/controller.
        """
        try:
            self._joystick = pygame.joystick.Joystick(joystick_index)
        except pygame.error:
            print(f"Cannot find Joystick/Controller at index {joystick_index}.")
            self._joystick = None
            return

        self._joystick.init()
        self._rumble_controller()
        print("Controller connected.")


    def read_inputs(self) -> CarCommands:
        car_commands = CarCommands()
        event = pygame.event.wait(100) #milliseconds

        # Check if the controller is still connected
        if event.type == pygame.JOYDEVICEREMOVED:
            self._joystick.quit()
            print("Controller disconnected. Stopping the car.")
            car_commands.stop = True
            self._joystick = None
            return car_commands

        if event.type == pygame.NOEVENT:
            return car_commands

        # Reconnection if the controller was disconnected
        if not self._joystick:
            print("Currently no controller/joystick connected, try reconnect...")
            self._connect_controller()
            car_commands.stop = True
            return car_commands

        self._clock.tick(self._input_freq)

        # Get throttle and check if forward or backward throttle is pressed
        forward_throttle = self._joystick.get_axis(5)
        backward_throttle = self._joystick.get_axis(2)
        if forward_throttle != -1 and backward_throttle == -1:
            car_commands.throttle = (forward_throttle+1)/2
        elif backward_throttle != -1 and forward_throttle == -1:
            car_commands.throttle = -((backward_throttle+1)/2)
        else:
            car_commands.throttle = 0

        car_commands.steer = self._joystick.get_axis(0)
        car_commands.stop = self._joystick.get_button(3)

        return car_commands
