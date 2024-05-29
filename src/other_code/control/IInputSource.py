from abc import ABC, abstractmethod

from CarCommands import CarCommands


class IInputSource(ABC):
    @abstractmethod
    def read_inputs(self) -> CarCommands:
        """
        Returns a list of commands based on the input source.
        Commands should be tuples in the format: ("command_type", command_data)
        """
        raise NotImplementedError

