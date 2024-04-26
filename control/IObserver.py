from abc import ABC, abstractmethod
class Observer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def update(self, frame):
        '''
        Called when an Observable notifies it's observers
        :return:
        '''
        raise NotImplementedError
