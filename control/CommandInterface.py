import serial
from threading import Lock

class CommandInterface:
    def __init__(self):
        port = "/dev/ttyS0"
        baudrate = 9600
        self.lock = Lock()
        # 8N1

        try:
            self.uart = serial.Serial(port, baudrate)
        except serial.serialutil.SerialException:
            print("could not open:", port)
            self.uart = None

    def sensor_toggle(self):
        with self.lock:
            print("toggled sensors", end="\r\n")
            if self.uart:
                self.uart.write(('toggleSensors\r').encode('utf8'))

    def stop(self):
        with self.lock:
            print("stop", end="\r\n")
            if self.uart:
                self.uart.write(('drive,0\r').encode('utf8'))
                self.uart.write(('steer,0\r').encode('utf8'))

    def throttle(self, value):
        with self.lock:
            value = value * 100
            #print("sending throttle " + str(value), end="\r\n")
            if self.uart:
                self.uart.write(('drive,' + str(int(value)) + '\r').encode('utf-8'))
                #print("sent throttle ")

    def steer(self, value):
        with self.lock:
            value = value * 100
            print("sending steer " + str(value), end="\r\n")
            if self.uart:
                num_bytes = self.uart.write(('steer,' + str(int(value)) + '\r').encode('utf-8'))
                print("sent steer, written bytes: ", num_bytes)
