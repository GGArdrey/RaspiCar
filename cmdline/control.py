#!/usr/bin/env python3

import sys
import tty
import termios

# python3-pyserial
import serial
# https://pyserial.readthedocs.io/en/latest/pyserial.html

class Control:

    def __init__(self):
        port = "/dev/ttyS0"
        baudrate = 115200
        # 8N1

        try:
            self.uart = serial.Serial(port, baudrate)
        except serial.serialutil.SerialException:
            print("could not open:", port)
            self.uart = None

        # uart.write()

        #uart.close()

    def sensor_toggle(self):
        print("toggled sensors", end="\r\n")
        if self.uart:
          self.uart.write(b'toggleSensors\r')    

    def forward(self):
        print("forward", end="\r\n")
        if self.uart:
          self.uart.write(b'drive,20\r')

    def backward(self):
        print("backward", end="\r\n")
        if self.uart:
            self.uart.write(b'drive,-20\r')

    def left(self):
        print("left", end="\r\n")
        if self.uart:
            self.uart.write(b'steer,-50\r')

    def right(self):
        print("right", end="\r\n")
        if self.uart:
            self.uart.write(b'steer,50\r')

    def stop(self):
        print("stop", end="\r\n")
        if self.uart:
            self.uart.write(b'drive,0\r')
            self.uart.write(b'steer,0\r')


def inputloop(ctrl):
    run = True

    # using terminal raw mode
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setraw(fd)


    while run:
        ch = sys.stdin.read(1)
        #ch = sys.stdin.read(3)

        ch = bytes(ch, encoding='utf8')
        #print(bytes(ch.encode()), end="\r\n")
        print(ch, end="\r\n")

        if ch == b' ':
            ctrl.stop()
        if ch == b'w':
            ctrl.forward()
        if ch == b's':
            ctrl.backward()
        if ch == b'a':
            ctrl.left()
        if ch == b'd':
            ctrl.right()
        if ch == b'!':
            ctrl.sensor_toggle()

        # Ctrl - c
        if ch == b"\x03":
            print("exit", end="\r\n")
            break


    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)




if __name__ == "__main__":
    control = Control()

    inputloop(control)


