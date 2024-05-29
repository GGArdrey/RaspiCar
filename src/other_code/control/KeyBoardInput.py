import pygame
from IInputSource import IInputSource
from CarCommands import CarCommands
class KeyboardInput(IInputSource):
    def __init__(self):
        pygame.init()
        self._keys = pygame.key.get_pressed()

    def _deinit(self):
        pygame.quit()

    def read_inputs(self) -> CarCommands:
        car_commands = CarCommands()

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                self._deinit()
                raise SystemExit

        self._keys = pygame.key.get_pressed()

        # Check for throttle
        if self._keys[pygame.K_w]:
            car_commands.throttle = 1.0
        elif self._keys[pygame.K_s]:
            car_commands.throttle = -1.0
        else:
            car_commands.throttle = 0.0

        # Check for steering
        if self._keys[pygame.K_a]:
            car_commands.steer = -1.0
        elif self._keys[pygame.K_d]:
            car_commands.steer = 1.0
        else:
            car_commands.steer = 0.0

        # Check for stop
        if self._keys[pygame.K_SPACE]:
            car_commands.stop = True
        else:
            car_commands.stop = False

        return car_commands