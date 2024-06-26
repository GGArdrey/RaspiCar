import csv
import os
import time
from datetime import datetime
from CarCommands import CarCommands
from CommandInterface import CommandInterface
from IControlAlgorithm import IControlAlgorithm
from IInputSource import IInputSource
from IObserver import IObserver
from threading import Thread, Lock, Condition

class ControlManager(IObserver):
    def __init__(self, command_interface, input_source, control_algorithm1, control_algorithm2):
        super().__init__()
        self.lock = Lock()
        self.new_data_condition = Condition(self.lock)
        self._command_interface: CommandInterface = command_interface
        self._input_source: IInputSource = input_source
        self._control_algorithm1: IControlAlgorithm = control_algorithm1
        self._control_algorithm2: IControlAlgorithm = control_algorithm2
        self.running = False
        self.thread = None

    def start(self):
        if not self.running:
            print("Starting Control Manager Thread...")
            self.running = True
            self.thread = Thread(target=self.run)
            self.thread.start()

    def stop(self):
        if self.running:
            self.running = False
            with self.new_data_condition:
                self.new_data_condition.notify()  # Wake up the thread if it's waiting
            if self.new_data_condition:
                self.thread.join()
            # Ensure clean shutdown, like stopping any motor or sending a stop command
            self._command_interface.stop()

    def update(self, frame, timestamp):
        with self.new_data_condition:
            self.new_data_condition.notify()  # Signal that a new frame is available



    def run(self):
        while self.running:
            with self.new_data_condition:
                self.new_data_condition.wait()  # Wait for a new frame
                input_commands = self._input_source.read_inputs() if self._input_source else None

                if not self._control_algorithm1 and not self._control_algorithm2:
                    self._execute_commands(input_commands)
                elif self._control_algorithm1 and not self._control_algorithm2:
                    algorithm1_commands = self._control_algorithm1.read_inputs()
                    merged_commands = self._merge_commands(input_commands, algorithm1_commands)
                    self._execute_commands(merged_commands)
                else:
                    raise NotImplementedError

    def _merge_commands(self, input_commands: CarCommands, algorithm_commands: CarCommands) -> CarCommands:
        merged_commands = CarCommands()
        merged_commands.throttle = input_commands.throttle if input_commands.throttle else algorithm_commands.throttle
        merged_commands.stop = input_commands.stop or algorithm_commands.stop
        merged_commands.steer = input_commands.steer if abs(input_commands.steer) > 0.15 else algorithm_commands.steer
        return merged_commands

    def _execute_commands(self, car_commands: CarCommands):
        # TODO maybe remake command interface to be more like _command_interface.execute(car_commands)
        if car_commands.stop:
            self._command_interface.stop()
        else:
            self._command_interface.steer(car_commands.steer)
            self._command_interface.throttle(car_commands.throttle)
