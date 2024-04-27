import time

from CarCommands import CarCommands
from CommandInterface import CommandInterface
from IControlAlgorithm import IControlAlgorithm
from IInputSource import IInputSource

from threading import Lock

class ControlManager:
    def __init__(self, command_interface, input_source, control_algorithm):
        self.lock = Lock()
        self._command_interface: CommandInterface = command_interface
        self._input_source: IInputSource = input_source
        self._control_algorithm: IControlAlgorithm = control_algorithm

    def run(self):
        while True:
            # Process input source
            input_commands = self._input_source.read_inputs()

            #print("Input commands:", input_commands.throttle, input_commands.steer, input_commands.stop)
            if self._control_algorithm is None:
                self._execute_commands(input_commands)
                continue
            else:
                # Process control algorithm
                algorithm_commands = self._control_algorithm.read_inputs()
                # Merge commands
                merged_commands = self._merge_commands(input_commands, algorithm_commands)
                #print("Command Interfaces: ", merged_commands.throttle, merged_commands.steer)
                #print("Algorithm Steering: ", algorithm_commands.throttle, algorithm_commands.steer)
                # Execute merged commands
                self._execute_commands(merged_commands)

    def _merge_commands(self, input_commands: CarCommands, algorithm_commands: CarCommands) -> CarCommands:
        merged_commands = CarCommands()

        # Use input source's throttle if anything applied
        if input_commands.throttle:
            merged_commands.throttle = input_commands.throttle
        else:
            merged_commands.throttle = algorithm_commands.throttle

        # Use stop from either one
        merged_commands.stop = input_commands.stop or algorithm_commands.stop

        # Use control algorithm's steering value if input source's steering is zero, otherwise use input source's
        # steering
        if input_commands.steer == 0.0:
            merged_commands.steer = algorithm_commands.steer
        else:
            merged_commands.steer = input_commands.steer

        return merged_commands

    def _execute_commands(self, car_commands: CarCommands):
        #with self.lock:
        if car_commands.stop:
            self._command_interface.stop()
        else:
            #print("execute steer: ", car_commands.steer)
            #print("execute throttle: ", car_commands.throttle)
            self._command_interface.steer(car_commands.steer)
            self._command_interface.throttle(car_commands.throttle)