
import serial
class DummyCommandInterface:
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


