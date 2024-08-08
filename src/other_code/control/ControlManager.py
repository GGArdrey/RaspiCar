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
        self.log_file_path = self.setup_csv("./log.csv")

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
                    self.process_algorithms(input_commands)

    def process_algorithms(self, input_commands):
        algorithm1_commands = self._control_algorithm1.read_inputs() if self._control_algorithm1 else None
        algorithm2_commands = self._control_algorithm2.read_inputs() if self._control_algorithm2 else None

        #[current_time, mse1, mse2, size1, size2, score1, score2]
        if algorithm2_commands and algorithm2_commands.additional_info and algorithm2_commands.additional_info[5] > 1 and algorithm2_commands.additional_info[6] > 1:  #
            algo = "PolyFit"
            merged_commands = self._merge_commands(input_commands, algorithm2_commands)
            if merged_commands.steer == input_commands.steer:
                algo = "MANUAL"
            print("\n\n USING " + algo + " \n\n")
            self.log_data(algo, merged_commands,
                          algorithm1_commands.additional_info if algorithm1_commands else [],
                          algorithm2_commands.additional_info if algorithm2_commands else [],
                          self.log_file_path)
            self._execute_commands(merged_commands)
        else:# algorithm1_commands and algorithm1_commands.additional_info and algorithm1_commands.additional_info[4] > 0.99: #4 is neihborhood prob
            algo = "PilotNet"
            merged_commands = self._merge_commands(input_commands, algorithm1_commands)
            if merged_commands.steer == input_commands.steer:
                algo = "MANUAL"
            print("\n\n USING " + algo + " \n\n")
            self.log_data(algo, merged_commands,
                          algorithm1_commands.additional_info if algorithm1_commands else [],
                          algorithm2_commands.additional_info if algorithm2_commands else [],
                          self.log_file_path)
            self._execute_commands(merged_commands)
        # Log data from both algorithms, regardless of which was used

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

    def log_data(self, algo_used, commands, additional_info1, additional_info2, file_path):
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            current_time = round(time.time() * 1000)

            if (algo_used == "PilotNet" or algo_used == "PolyFit" or algo_used == "MANUAL") and additional_info1 and additional_info2:
                row = [current_time, algo_used, commands.steer, commands.throttle] + additional_info1 + additional_info2
            else:
                # If no additional info available, log basic data with placeholders
                row = [current_time, algo_used, commands.steer, commands.throttle] + \
                      ["N/A"] * 13
            writer.writerow(row)

    def setup_csv(self, file_path):
        # Define the headers for the CSV file
        headers = [
            "system_timestamp", "algorithm_used", "steer_executed", "throttle_executed",
            "pilotnet_timestamp", "predicted_steer", "weighted_steer", "max_prob",
            "neighborhood_prob", "nn_predictions","polyfit_timestamp", "lane1_mse", "lane2_mse", "lane1_cluster_size",
            "lane2_cluster_size", "lane1_score", "lane2_score"
        ]

        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

        return file_path