from abc import ABC
from IObserver import IObserver
import threading

class IObservable:
    def __init__(self):
        self._observers = []
        self._observers_lock = threading.Lock()

    def register_observer(self, observer):
        with self._observers_lock:
            if observer not in self._observers:
                self._observers.append(observer)

    def remove_observer(self, observer):
        with self._observers_lock:
            self._observers.remove(observer)

    def _notify_observers(self, data, timestamp):
        with self._observers_lock:
            for observer in self._observers:
                observer.update(data, timestamp)
