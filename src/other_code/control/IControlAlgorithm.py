from abc import ABC, abstractmethod

from CarCommands import CarCommands
from IObserver import IObserver


class IControlAlgorithm(IObserver):
    @abstractmethod
    def process_frame(self, frame):
        """
        Returns a list of commands based on the control algorithm's output.
        Commands should be tuples in the format: ("command_type", command_data)
        """
        raise NotImplementedError

    @abstractmethod
    def read_inputs(self) -> CarCommands:
        raise NotImplementedError

    @abstractmethod
    def update(self, data, timestamp):
        raise NotImplementedError

