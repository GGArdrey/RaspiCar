'''
XBOX BUTTON MAPPINGS:
A -> button 0
B -> button 1
X -> button 2
Y -> button 3

Right Joystick:
left-right -> axis 3 (neutral 0; left -1; right 1)
up-down -> axis 4 (neutral 0; up -1; down 1)
click -> button 10

Left Joystick:
left-right -> axis 0 (neutral 0; left -1; right 1)
up-down -> axis 1 (neutral 0; up -1; down 1)
click -> button 9

start -> button 7
back -> button 6

D-Pad:
up -> button 13
down -> button 12
left -> button 14
right -> button 11

Right Shoulder
top -> button 5
bottom -> axis 5 (default -1; fully pressed 1)

Left Shoulder
top -> button 4
bottom -> axis 2 (default -1; fully pressed 1)

'''
import time

import pygame

class JoystickHandler:
    def __init__(self, joystick_index=0, freq_Hz = 10):
        self.freq_Hz = freq_Hz
        pygame.init()
        self.clock = pygame.time.Clock()
        try:
            self.joystick = pygame.joystick.Joystick(joystick_index)
        except pygame.error as message:
            print("Cannot find Joystick/Controller. Please connect one and restart program")
            raise SystemExit(message)
        self.joystick.init()
        self.joystick.rumble(low_frequency=1.0, high_frequency=1.0, duration=200)
        time.sleep(0.2)
        self.joystick.rumble(low_frequency=1.0, high_frequency=1.0, duration=200)

    def read_inputs(self):
        event = pygame.event.wait()

        # Check if Controller disconneted in the meantime
        if event.type == pygame.JOYDEVICEREMOVED:
            self.joystick.quit()
            raise Exception('A Joystick/Controller Disconnected')

        # Wait for reconnection
        while not (self.joystick.get_init()):
            event = pygame.event.wait()
            if event.type == pygame.JOYDEVICEADDED:
                self.joystick.init()
                self.joystick.rumble(low_frequency=1.0,high_frequency=1.0, duration=200)
                time.sleep(200)
                self.joystick.rumble(low_frequency=1.0, high_frequency=1.0, duration=200)
                print("Joystick added")


        self.clock.tick(self.freq_Hz)
        #pygame.event.pump()  # Process event queue

        forward = self.joystick.get_axis(2)
        backward = self.joystick.get_axis(5)
        sensor_toggle = self.joystick.get_button(7)
        steering = self.joystick.get_axis(0)
        stop = self.joystick.get_button(3)

        return forward, backward, sensor_toggle, steering, stop

    def rumble_joystick(self):
        self.joystick.rumble(low_frequency=1.0,high_frequency=1.0, duration=500)

    def cleanup(self):
        pygame.quit()



#!/usr/bin/env python3

import serial
# https://pyserial.readthedocs.io/en/latest/pyserial.html

class CarController:

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
        print("throttle " + str(int(value)), end="\r\n")
        if self.uart:
          self.uart.write(bytes('drive,' + str(int(value)) + '\r',encoding='utf8'))

    def steer(self, value):
        print("steer " + str(int(value)), end="\r\n")
        if self.uart:
            self.uart.write(bytes('steer,'+ str(int(value)) + '\r',encoding='utf8'))





NO_THROTTLE = -1
PRESSED = 1.0
def inputloop(control):
    joystickHandler = JoystickHandler()

    while(True):
        # Read state of xbox controller buttons/joystick
        try:
            forward, backward, sensor_toggle, steering, stop = joystickHandler.read_inputs()
        except Exception as e:
            control.stop()
            print(e)


       # execute commands based on button states
        if stop == PRESSED:
            control.stop()
        if sensor_toggle == PRESSED:
            control.sensor_toggle()

        control.steer(steering*100) #conversion

        if forward != NO_THROTTLE and backward != NO_THROTTLE: # If forward and backward throttle is pressed, do nothing
            control.throttle(0)
        elif forward == NO_THROTTLE and backward == NO_THROTTLE: # or when both are not pressed
            control.throttle(0)
        elif forward != NO_THROTTLE: # if only forward is pressed
            control.throttle((forward + 1) / 2 * 100)
        elif backward != NO_THROTTLE: # if only backward is pressed
            control.throttle(-(backward + 1) / 2 * 100)





if __name__ == "__main__":
    control = CarController()
    inputloop(control)


