"""
RaspiCar
Copyright (c) 2024 Fynn Luca Maaß

Licensed under the Custom License. See the LICENSE file in the project root for license terms.
"""

class DummyCommandInterface:
    '''
    This is a dummy CommandInterface that can be used instead of the actual one. It will not send UART to the Pico,
    instead it just prints everything to the console. Maybe use it for testing/debugging.
    '''
    def __init__(self):
        pass
    def sensor_toggle(self):
        print("toggled sensors", end="\r\n")


    def stop(self):
        print("stop", end="\r\n")


    def throttle(self, value):
        value = value * 100
        print("throttle " + str(value), end="\r\n")


    def steer(self, value):
        value = value*100
        print("steer " + str(value), end="\r\n")


