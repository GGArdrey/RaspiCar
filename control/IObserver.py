from abc import ABC, abstractmethod


class IObserver(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def update(self, frame, timestamp):
        '''
        Called when an Observable notifies it's observers
        :return:
        '''
        raise NotImplementedError
