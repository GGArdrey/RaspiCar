class CarCommands:
    def __init__(self, steer=0.0, throttle=0.0, stop=0.0, start_capture=False, stop_capture=False):
        self._steer = steer
        self._throttle = throttle
        self._stop = stop #TODO this needs to be bool not float
        self._start_capture = start_capture
        self._stop_capture = stop_capture

    def copy(self):
        '''
        Create a copy of itself
        :return: CarCommands
        '''
        return CarCommands(self._steer, self._throttle, self._stop, self.start_capture, self.stop_capture)

    @property
    def steer(self) -> float:
        return self._steer

    @steer.setter
    def steer(self, value: float):
        if not isinstance(value, (float, int)):  # Allow int as well, since they can be safely converted to float.
            raise TypeError("Steer value must be a float or int")
        self._steer = float(value)

    @property
    def throttle(self) -> float:
        return self._throttle

    @throttle.setter
    def throttle(self, value: float):
        if not isinstance(value, (float, int)):  # Same allowance for int.
            raise TypeError("Throttle value must be a float or int")
        self._throttle = float(value)

    @property
    def stop(self) -> float:
        return self._stop

    @stop.setter
    def stop(self, value: float):
        if not isinstance(value, (float, int)):
            raise TypeError("Stop value must be a float or int")
        self._stop = float(value)

    @property
    def start_capture(self) -> bool:
        return self._start_capture

    @start_capture.setter
    def start_capture(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("Start capture value must be a boolean")
        self._start_capture = value

    @property
    def stop_capture(self) -> bool:
        return self._stop_capture

    @stop_capture.setter
    def stop_capture(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("Stop capture value must be a boolean")
        self._stop_capture = value
