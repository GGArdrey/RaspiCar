import serial
import time
import threading
from threading import Lock

class CommandInterface:
    def __init__(self):
        port = "/dev/ttyS0"
        baudrate = 115200
        # 8N1
        try:
            self.uart = serial.Serial(port, baudrate)
            self.lock = Lock()
            self.ping_thread = threading.Thread(target=self.send_ping)
            self.ping_thread.daemon = True
            self.ping_thread.start()
        except serial.serialutil.SerialException:
            print("could not open:", port)
            self.uart = None

    def send_ping(self):
        while True:
            if self.uart:
                with self.lock:
                    self.uart.write(('ping,1\n').encode('utf8'))  # Send the "ping" command
            time.sleep(0.1)  # Wait for 100ms

    def sensors_enable(self):
        if self.uart:
            with self.lock:
                self.uart.write(('sensors_enable,1\n').encode('utf8'))

    def sensors_disable(self):
        if self.uart:
            with self.lock:
                self.uart.write(('sensors_disable,1\n').encode('utf8'))

    def emergency_stop(self):
        if self.uart:
            with self.lock:
                self.uart.write(('emergency_stop,1\n').encode('utf8'))

    def reset_emergency_stop(self):
        if self.uart:
            with self.lock:
                self.uart.write(('reset_emergency_stop,1\n').encode('utf8'))

    def throttle(self, value):
        value = value * 100
        if self.uart:
            with self.lock:
                self.uart.write(('drive,' + str(int(value)) + '\n').encode('utf-8'))

    def steer(self, value):
        value = value * 100
        if self.uart:
            with self.lock:
                self.uart.write(('steer,' + str(int(value)) + '\n').encode('utf-8'))

# Example usage
if __name__ == "__main__":
    ci = CommandInterface()
