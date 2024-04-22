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
        self._steer = value

    @property
    def throttle(self) -> float:
        return self._throttle

    @throttle.setter
    def throttle(self, value: float):
        self._throttle = value

    @property
    def stop(self) -> float:
        return self._stop

    @stop.setter
    def stop(self, value: float):
        self._stop = value