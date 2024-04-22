
import serial
class CommandInterface:
    def __init__(self):
        port = "/dev/ttyS0"
        baudrate = 115200
        # 8N1

        try:
            self.uart = serial.Serial(port, baudrate)
        except serial.serialutil.SerialException:
            print("could not open:", port)
            self.uart = None

    def sensor_toggle(self):
        print("toggled sensors", end="\r\n")
        if self.uart:
          self.uart.write(b'toggleSensors\r')

    def stop(self):
        print("stop", end="\r\n")
        if self.uart:
            self.uart.write(b'drive,0\r')
            self.uart.write(b'steer,0\r')

    def throttle(self, value):
        value = value * 100
        #print("throttle " + str(value), end="\r\n")
        if self.uart:
          self.uart.write(bytes('drive,' + str(int(value)) + '\r',encoding='utf8'))

    def steer(self, value):
        value = value*100
        #print("steer " + str(value), end="\r\n")
        if self.uart:
            self.uart.write(bytes('steer,'+ str(int(value)) + '\r',encoding='utf8'))

