import machine
import time
from vl53l0x import VL53L0X
from machine import Timer
from FiniteStateMachine import FiniteStateMachine

def range_mapping(x, in_min, in_max, out_min, out_max):
    val = (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    return int(val)


class CommunicationManager:
    def __init__(self, car_instance):
        self.uart = machine.UART(0, 115200)
        self.uart.init(115200, bits=8, parity=None, stop=1, timeout=100)
        self.car = car_instance
        self.fsm = FiniteStateMachine(car_instance)
        self.ping_alive = False

        # This defines all commands that are received over UART
        self.command_map = {
            "ping": self.ping_success,
            "drive": self.drive,
            "steer": self.steer,
            "sensors_enable": self.enable_sensors,
            "sensors_disable": self.disable_sensors,
            "emergency_stop": self.manual_emergency,
            "reset_emergency_stop": self.reset_manual_emergency
        }

    def process_commands(self):
        while True:
            line = self.read_line_from_uart()
            print("received command: ", line)
            if not line:
                continue
            command, *args = line.split(",")
            self.execute_command(command, args)

    def execute_command(self, command, args):
        if command in self.command_map:
            self.command_map[command](args)
        else:
            print(f"Invalid command: {command}")

    def read_line_from_uart(self):
        buffer = bytearray()
        try:
            while True:
                char = self.uart.read(1)
                if char:
                    if char in [b'\n', b'\r']:
                        break
                    buffer += char
            return buffer.decode('utf-8').strip() if buffer else ''
        except Exception as e:
            print(f"UART read error: {e}")
            return ''

    def drive(self, args):
        self.fsm.handle_event('drive', *args)

    def steer(self, args):
        self.fsm.handle_event('steer', *args)

    def enable_sensors(self, args):
        self.car.stop_sensors(self.read_sensors) #TODO this bypasses fsm
        self.fsm.handle_event('enable_sensors', *args)

    def disable_sensors(self, args):
        self.car.stop_sensors() #TODO this bypasses fsm
        self.fsm.handle_event('disable_sensors', *args)

    def manual_emergency(self, args):
        self.fsm.handle_event('manual_emergency', *args)

    def reset_manual_emergency(self, args):
        self.fsm.handle_event('reset_manual_emergency', *args)

    def ping_success(self, args):
        self.ping_alive = True
        self.fsm.handle_event('ping_success', *(args or []))

    def ping_timeout(self, args):
        self.fsm.handle_event('ping_timeout', *(args or []))

    def read_sensors(self, timer):
        front_distance = self.car.measure_front_distance()
        rear_distance = self.car.measure_rear_distance()
        self.fsm.handle_event('front_tof_measurement', front_distance)
        self.fsm.handle_event('rear_tof_measurement', rear_distance)
        print("Front distance: ", front_distance, "Rear distance: ", rear_distance)

    def watchdog_callback(self, timer):
        # TODO maybe move ing_sucess() and ping_timeout() in one ping() function and use this ping() function
        #  as callback instead of this. Like it is with read sensors
        if not self.ping_alive:
            self.ping_timeout(None)
        self.ping_alive = False


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

        self.i2c_back = machine.I2C(1, scl=machine.Pin(3), sda=machine.Pin(2))
        self.i2c_front = machine.I2C(0, scl=machine.Pin(5), sda=machine.Pin(4))

        self.tof_front = VL53L0X(self.i2c_front)
        self.tof_back = VL53L0X(self.i2c_back)

        self.sensor_timer = Timer(-1)
        self.watchdog_timer = Timer(-1)
        self.led_timer = Timer(-1)

    def motor_drive(self, in_speed):
        if not (-100 <= in_speed <= 100):
            print("Warning: motor_drive: in_speed must be between -100 and 100")
            in_speed = max(min(in_speed, 100), -100)

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
        if not (-100 <= in_amount <= 100):
            print("Warning: servo_steer: in_amount must be between -100 and 100")
            in_amount = max(min(in_amount, 100), -100)

        amount = range_mapping(in_amount, -100, 100, self.servo_max_left, self.servo_max_right)
        self.servo_pwm.duty_u16(max(min(amount, self.servo_max_right), self.servo_max_left))

    def start_sensors(self, callback):
        self.sensor_timer.init(period=100, mode=Timer.PERIODIC, callback=callback)

    def stop_sensors(self):
        self.sensor_timer.deinit()

    def measure_front_distance(self):
        return self.tof_front.ping()

    def measure_rear_distance(self):
        return self.tof_back.ping()

    def start_watchdog(self, callback):
        self.watchdog_timer.init(period=200, mode=Timer.PERIODIC, callback=callback)

    def stop_watchdog(self):
        self.watchdog_timer.deinit()

    def flash_onboard_led(self, freq):
        period = int(1000 / freq)

        def toggle_led(timer):
            self.led.value(not self.led.value())

        # Start the timer to toggle the LED
        self.led_timer.init(period=period, mode=Timer.PERIODIC, callback=toggle_led)

    def stop_onboard_led(self):
        self.led_timer.deinit()
        self.led.value(0)  # Turn off LED

if __name__ == "__main__":
    picar = Car()
    comm_manager = CommunicationManager(picar)

    # Start watchdog timer with the callback to comm_manager
    picar.start_watchdog(comm_manager.watchdog_callback)

    # Start sensor reading with the callback to comm_manager
    picar.start_sensors(comm_manager.read_sensors)

    time.sleep(1)
    comm_manager.process_commands()