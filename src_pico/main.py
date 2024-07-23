import machine
import time
from vl53l0x import VL53L0X
from machine import Timer


def range_mapping(x, in_min, in_max, out_min, out_max):
    val = (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    return int(val)


class Distance:
    direction = "f"
    debug_flag = False

    def __init__(self):
        print("Initializing distance sensors")
        self.i2c_back = machine.I2C(1, scl=machine.Pin(3), sda=machine.Pin(2))
        self.i2c_front = machine.I2C(0, scl=machine.Pin(5), sda=machine.Pin(4))

        print("Scanning I2C bus...")
        print(self.i2c_front.scan())
        time.sleep(1)

        self.tof_front = VL53L0X(self.i2c_front)
        self.tof_back = VL53L0X(self.i2c_back)

    def measure(self):
        sensor = self.tof_front if self.direction == "f" else self.tof_back
        if sensor:
            data = sensor.ping()
            if self.debug_flag:
                print(f"{'Front' if self.direction == 'f' else 'Back'} Sensor: ", data)
            return data


class Comm:
    def __init__(self, car_inst, dist_inst):
        self.uart = machine.UART(0, 115200)  # init with given baudrate
        self.uart.init(115200, bits=8, parity=None, stop=1, timeout=100)  # init with given parameters 115200

        # self.uart = sys.stdin

        self.range_min = -100
        self.range_max = 100
        self.car = car_inst
        self.sensor = dist_inst

        self.commands = {"drive": self.drive,
                         "steer": self.steer,
                         "toggleSensors": self.toggleSensors,
                         "stop": self.stop}

    def readline(self):
        buffer = bytearray()
        try:
            while True:
                char = self.uart.read(1)  # Read one byte at a time
                if char:
                    # Check if the end of a line has been reached
                    if char == b'\n' or char == b'\r':  # Assuming newline or carriage return ends the line
                        break
                    buffer += char

            # If we exit the loop, we attempt to decode what we have
            if buffer:
                line = buffer.decode('utf-8').strip('\r\n')
                print("Received line:", line)
                return line
            else:
                print("Received an empty line or just newline")
                return ''
        except Exception as e:
            print("Error during UART read or decode:", str(e))
            return ''

    def process(self):
        while True:
            line = self.readline()
            print("received input:", line)
            if not line:
                continue
            line = line.strip()
            if not line:
                continue
            inst = line.split(",")
            command = inst[0]
            func = self.commands.get(command, None)
            if not func:
                print("invalid command: ", command)
                continue
            args = inst[1:]
            func(args)

    def stop(self, args):
        self.car.motor_drive(0)
        self.car.servo_steer(0)

    def toggleSensors(self, args):
        if self.sensor.debug_flag:
            self.sensor.debug_flag = False
            return
        self.sensor.debug_flag = True

    def drive(self, args):
        print("Driving", args)
        if not len(args) == 1:
            print("Invalid parameters for \"drive\", speed (signed integer).")
            return

        val = int(args[0])
        if val < -100:
            val = -100
        if val > 100:
            val = 100

        self.car.motor_drive(val)

    def steer(self, args):
        if not len(args) == 1:
            print("Invalid parameters for \"steer\", expecting direction (signed integer) between -100 and + 100 %.")
            return
        val = int(args[0])
        print("set servo to ", val)
        self.car.servo_steer(val)


class Car:
    def __init__(self):
        self.servo_neutral = 4500
        self.servo_max_left = 2500
        self.servo_max_right = 6500
        self.servo_frequency = 50
        self.motor_frequency = 15000
        self.direction = ""
        self.led = machine.Pin(25, machine.Pin.OUT)

        self.servo_pwm = machine.PWM(machine.Pin(15, machine.Pin.OUT))
        self.servo_pwm.freq(self.servo_frequency)
        self.servo_pwm.duty_u16(self.servo_neutral)

        self.motor_A_enable = machine.Pin(21, machine.Pin.OUT)
        self.motor_B_enable = machine.Pin(16, machine.Pin.OUT)

        self.motor_A_pwm = machine.PWM(self.motor_A_enable)
        self.motor_B_pwm = machine.PWM(self.motor_B_enable)
        self.motor_A_pwm.freq(self.motor_frequency)
        self.motor_B_pwm.freq(self.motor_frequency)

        self.motor_A_pwm.duty_u16(0)
        self.motor_B_pwm.duty_u16(0)

        self.motor_A_F = machine.Pin(20, machine.Pin.OUT)
        self.motor_A_B = machine.Pin(19, machine.Pin.OUT)
        self.motor_B_F = machine.Pin(18, machine.Pin.OUT)
        self.motor_B_B = machine.Pin(17, machine.Pin.OUT)
        self.motor_A_F.off()
        self.motor_A_B.off()
        self.motor_B_F.off()
        self.motor_B_B.off()

    def motor_drive(self, in_speed):
        speed = range_mapping(abs(in_speed), 0, 100, 50001, 74999) if in_speed != 0 else 0
        new_direction = "backward" if in_speed < 0 else "forward"

        if new_direction != self.direction:
            if new_direction == "forward":
                self.motor_A_B.off()
                self.motor_B_B.off()
                self.motor_A_F.on()
                self.motor_B_F.on()
            else:
                self.motor_A_F.off()
                self.motor_B_F.off()
                self.motor_A_B.on()
                self.motor_B_B.on()

            self.direction = new_direction

        self.motor_A_pwm.duty_u16(speed)
        self.motor_B_pwm.duty_u16(speed)

    def servo_steer(self, in_amount):
        amount = range_mapping(in_amount, -100, 100, self.servo_max_left, self.servo_max_right)
        self.servo_pwm.duty_u16(max(min(amount, self.servo_max_right), self.servo_max_left))


if __name__ == "__main__":
    led = machine.Pin(25, machine.Pin.OUT)

    picar = Car()

    comm = Comm(picar, None)  # distance

    time.sleep(1)

    comm.process()





