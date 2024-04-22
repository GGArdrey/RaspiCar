from abc import ABC, abstractmethod

from CarCommands import CarCommands


class IControlAlgorithm(ABC):
    @abstractmethod
    def process_frame(self) -> CarCommands:
        """
        Returns a list of commands based on the control algorithm's output.
        Commands should be tuples in the format: ("command_type", command_data)
        """
        raise NotImplementedError