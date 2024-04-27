from abc import ABC
from IObserver import IObserver
class IObservable(ABC):
    def __init__(self):
        self._observers = []

    def register_observer(self, observer : IObserver):
        self._observers.append(observer)

    def _notify_observers(self, data):
        for observer in self._observers:
            observer.update(data)