class CarCommands:
    def __init__(self):
        self._steer = 0.0
        self._throttle = 0.0
        self._stop = 0.0

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