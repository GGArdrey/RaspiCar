
import machine
import time
import sys
from vl53l0x import VL53L0X
from machine import Timer

def range_mapping(x, in_min, in_max, out_min, out_max):
        val = (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
        return int(val)


class AliveTimer:

    def __init__(self, comm, ms):
        self.ms = ms
        self.comm = comm
        self.alive = True
        self.comm.alivetimer = self
        self.KAtoggle = False

    def start(self):
        self.timer = Timer(
            period=self.ms,
            mode=Timer.PERIODIC,
            callback=self.callback
        )

    def stop(self):
        self.timer.deinit()

    def callback(self, timer):
        self.comm.car.led.toggle()
        self.comm.uart.write("keepalive callback\r\n")
        if self.KAtoggle:
            if not self.alive:
                print("Keep Alive Failed...")
                print("Stopping motors...")
                self.comm.car.motor_drive("A", 0)
                self.comm.car.motor_drive("B", 0)
                return

        self.alive = False    
        


class DistanceTimer:

    def __init__(self, comm, ms):
        self.ms = ms
        self.comm = comm
        self.mindist = 500
        self.comm.disttimer = self

    def start(self):
        self.timer = Timer(
            period=self.ms,
            mode=Timer.PERIODIC,
            callback=self.callback
        )

    def stop(self):
        self.timer.deinit()

    def callback(self, timer):
        dist = self.comm.sensor.measure()
        if dist:
            if dist < self.mindist:
                print("Distance Sensor found distance under min value: ", dist)
                print("Stopping motors...")
                print("Direction: ", self.comm.sensor.direction)
                decel_counter = 100
                if self.comm.sensor.direction == "f":
                    self.comm.drive([-decel_counter])
                    print("Backwards counter-acceleraion ...")
                elif self.comm.sensor.direction == "b":
                    self.comm.drive([decel_counter])
                    print("Forwards counter-acceleraion ...")
                self.comm.stop([0])
                self.comm.car.stop_flag = True

                #self.comm.uart.write("distance callback\r\n")
            else:
                self.comm.car.stop_flag = False


class Distance:

    direction = "f"
    debug_flag = False

    def __init__(self):
        print("init distance sensors")

        #TODO check forward backward only read forward sensor forward and backward sensor backward a.s.o

        #self.i2c = machine.I2C(1, 27, 26)


        scl_back = machine.Pin(3)
        sda_back = machine.Pin(2)
        self.i2c_back = machine.I2C(1, scl=scl_back, sda=sda_back)
        scl_front = machine.Pin(5)
        sda_front = machine.Pin(4)
        self.i2c_front = machine.I2C(0, scl=scl_front, sda=sda_front)

        #self.i2c.init(scl, sda)

        print("scanning bus...")
        devices = self.i2c_front.scan()
        print(devices)
        time.sleep(1)
        # try:
        self.tof_front = VL53L0X(self.i2c_front)
        self.tof_back = VL53L0X(self.i2c_back)
        # except OSError:
        #     print("No Sensor")
        #     self.tof = None
            

    def measure(self):
        if self.direction == "f":
            if not self.tof_front:
                return
            data = self.tof_front.ping()
            if self.debug_flag == True: 
                print("Front Sensor: ", data)
            return data
        elif self.direction == "b":
            if not self.tof_back:
                return
            data = self.tof_back.ping()
            if self.debug_flag == True: 
                print("Back Sensor: ", data)
            return data
        else:
            return


class Comm:
    def __init__(self, car_inst, dist_inst):
        self.uart = machine.UART(0, 115200)                          # init with given baudrate 
        self.uart.init(115200, bits=8, parity=None, stop=1)          # init with given parameters
        
        #self.uart = sys.stdin

        self.range_min = -100
        self.range_max = 100
        self.car = car_inst
        self.sensor = dist_inst
        self.alivetimer = None
        self.disttimer = None


        self.commands = {"drive": self.drive,
                        "steer": self.steer,
                        "toggleSensors": self.toggleSensors,
                        "keepalive": self.keepalive,
                        "check": self.check,
                        "servo_check": self.check_servo,
                        "motor_check": self.check_motor,
                        "toggleKA": self.toggleKA,
                        "drivef":self.drivef,
                        "driveb":self.driveb,
                        "stop":self.stop}


    def readline(self):
        buf = ""
        while True:
            char = self.uart.read(1)
            if not char:
                continue
            #print("raw buffer:", buf)
            if char.decode() == "\r":
                break
            buf += char.decode()
        #buf = buf.strip()
        return buf


    def process(self):
        while True:
            line = self.readline()
            print("got input:", line)
            if not line:
                continue
            line = line.strip()
            if not line:
                #distance.measure()
                continue
            inst = line.split(",")
            #print("split cmd:", inst)
            command = inst[0]
            func = self.commands.get(command, None)
            if not func:
                print("invalid command: ", func)
                continue
            args = inst[1:]
            func(args)

    def stop(self,args):
        self.car.motor_drive("A", 0)
        self.car.motor_drive("B", 0)
        self.car.servo_steer(0)

    def toggleSensors(self, args):
        if self.sensor.debug_flag:
            self.sensor.debug_flag = False
            return
        self.sensor.debug_flag = True
        

    def drivef(self, args):
        self.drive([100]) 
        time.sleep(0.1)
        self.drive([10])    
        self.sensor.direction = "f"  

    def driveb(self, args):
        self.drive([-100]) 
        time.sleep(0.1)
        self.drive([-10])  
        self.sensor.direction = "b"    

    def toggleKA(self, args):
        if self.alivetimer.KAtoggle:
            self.alivetimer.KAtoggle = False
        else:
            self.alivetimer.KAtoggle = True
            
    def keepalive(self, args):
        self.alivetimer.alive = True

    def check(self, args):
        print(args)
        self.car.perform_full_check()

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
        
        

        if val >= 0:
            self.sensor.direction = "f"
            if self.car.stop_flag:
                val = 0
                return
            self.car.motor_drive("A", val)
            self.car.motor_drive("B", val)
        if val < 0:
            self.sensor.direction = "b"
            if self.car.stop_flag:
                val = 0
                return
            self.car.motor_drive("A", val)
            self.car.motor_drive("B", val)

    def steer(self, args):
        if not len(args) == 1:
            print("Invalid parameters for \"steer\", expecting direction (signed integer) between -100 and + 100 %.")
            return
        val = int(args[0])
        print("set servo to ", val)
        self.car.servo_steer(val)

    def check_servo(self, args):
        print("Servo Checking", args)    
        self.car.servo_check()

    def check_motor(self, args):
        print("Servo Checking", args)    
        self.car.motor_check()
    

class Car:
    def __init__(self):
        self.servo_neutral = 4500
        self.servo_max_left = 2500
        self.servo_max_right = 6500
        self.servo_frequency = 50
        self.motor_frequency = 15000
        self.stop_flag = False
        self.direction = ""
        self.led = machine.Pin(25, machine.Pin.OUT)
        # init pins

        # servo
        self.servo = machine.Pin(15, machine.Pin.OUT)
        self.servo_pwm = machine.PWM(self.servo)
        self.servo_pwm.freq(self.servo_frequency)
        self.servo_pwm.duty_u16(self.servo_neutral)

        # H-bridge
        self.motor_A_enable = machine.Pin(21, machine.Pin.OUT)
        self.motor_B_enable = machine.Pin(16, machine.Pin.OUT)

        self.motor_A_pwm = machine.PWM(self.motor_A_enable)
        self.motor_B_pwm = machine.PWM(self.motor_B_enable)

        self.motor_A_pwm.freq(self.motor_frequency)
        self.motor_B_pwm.freq(self.motor_frequency)

        self.motor_A_pwm.duty_u16(0)
        self.motor_B_pwm.duty_u16(0)

        # H-bridge motor A directions
        self.motor_A_F = machine.Pin(20, machine.Pin.OUT)
        self.motor_A_B = machine.Pin(19, machine.Pin.OUT)
        self.motor_A_F.off()
        self.motor_A_B.off()

        # H-bridge motor B directions
        self.motor_B_F = machine.Pin(18, machine.Pin.OUT)
        self.motor_B_B = machine.Pin(17, machine.Pin.OUT)
        self.motor_B_F.off()
        self.motor_B_B.off()

    def motor_drive(self, motor, in_speed):
        speed = 0
        if in_speed < 0:
            speed = range_mapping(in_speed, 0, -100, 50001, 74999)
            self.direction = "backward"
        if in_speed > 0:
            speed = range_mapping(in_speed, 0, 100, 50001, 74999)
            self.direction = "forward"
        if in_speed == 0:
            speed = 0
            self.direction = "forward"

        if motor == "A":
            if self.direction == "forward":
                self.motor_A_F.off()
                self.motor_A_B.off()
                self.motor_A_F.toggle()
                self.motor_A_pwm.duty_u16(speed)

            elif self.direction == "backward":
                self.motor_A_F.off()
                self.motor_A_B.off()
                self.motor_A_B.toggle()
                self.motor_A_pwm.duty_u16(speed)
            else:
                print("ERROR: Function needs direction to be either 'forward' or 'backward'")

        elif motor == "B":
            if self.direction == "forward":
                self.motor_B_F.off()
                self.motor_B_B.off()
                self.motor_B_F.toggle()
                self.motor_B_pwm.duty_u16(speed)

            elif self.direction == "backward":
                self.motor_B_F.off()
                self.motor_B_B.off()
                self.motor_B_B.toggle()
                self.motor_B_pwm.duty_u16(speed)
        else:
            print("ERROR: Function needs id to be either 'A' or 'B'.")

    def servo_steer(self, in_amount):
        amount = range_mapping(in_amount, -100, 100, self.servo_max_left, self.servo_max_right)
        if amount < self.servo_max_left:
            amount = self.servo_max_left
        elif amount > self.servo_max_right:
            amount = self.servo_max_right

        self.servo_pwm.duty_u16(amount)


    def perform__full_check(self):
        #Testing Servo
        print("Testing servo...")

        print("  center")
        self.servo_steer(0) # ~ Center position
        time.sleep(1)
        print("  left")
        self.servo_steer(-100) # ~max left
        time.sleep(1)
        print("  right")
        self.servo_steer(100) # ~max right
        time.sleep(1)
        print("  center")
        self.servo_steer(0) # ~ Center position
        time.sleep(1)

        #return

        #Testing Motors
        print("Testing motors...")

        self.motor_drive("A", 0)
        self.motor_drive("B", 0)

        print("  right fwd")
        self.motor_drive("A", 100)
        time.sleep(1)
        self.motor_drive("A", 0)
        time.sleep(0.5)

        print("  left fwd")
        self.motor_drive("B", 100)
        time.sleep(1)
        self.motor_drive("B", 0)
        time.sleep(0.5)

        print("  right back")
        self.motor_drive("A", -100)
        time.sleep(1)
        self.motor_drive("A", 0)
        time.sleep(0.5)

        print("  left back")
        self.motor_drive("B", -100)
        time.sleep(1)
        self.motor_drive("B", 0)
        time.sleep(0.5)


        print("  both fwd slow")
        # start fast
        self.motor_drive("A", 100)
        self.motor_drive("B", 100)
        time.sleep(0.1)
        self.motor_drive("A", 50)
        self.motor_drive("B", 50)
        time.sleep(2)
        self.motor_drive("A", 0)
        self.motor_drive("B", 0)
        time.sleep(0.5)


        print("DONE")

    def servo_check(self):
        print("Performing Servo Test:")
        print("Centering Servo...")
        self.servo_steer(0)
        time.sleep(1)
        self.servo_steer(-100)
        print("Full Left...")
        time.sleep(1)
        print("Iterating from 0 to -100 in increments of 10")
        self.servo_steer(0)
        time.sleep(0.5)
        self.servo_steer(-10)
        time.sleep(0.5)
        self.servo_steer(-20)
        time.sleep(0.5)
        self.servo_steer(-30)
        time.sleep(0.5)
        self.servo_steer(-40)
        time.sleep(0.5)
        self.servo_steer(-50)
        time.sleep(0.5)
        self.servo_steer(-60)
        time.sleep(0.5)
        self.servo_steer(-70)
        time.sleep(0.5)
        self.servo_steer(-80)
        time.sleep(0.5)
        self.servo_steer(-90)
        time.sleep(0.5)
        self.servo_steer(-100)
        time.sleep(1)

        self.servo_steer(100)
        print("Full Right...")
        time.sleep(1)
        print("Iterating from 0 to 100 in increments of 10")
        self.servo_steer(0)
        time.sleep(0.5)
        self.servo_steer(10)
        time.sleep(0.5)
        self.servo_steer(20)
        time.sleep(0.5)
        self.servo_steer(30)
        time.sleep(0.5)
        self.servo_steer(40)
        time.sleep(0.5)
        self.servo_steer(50)
        time.sleep(0.5)
        self.servo_steer(60)
        time.sleep(0.5)
        self.servo_steer(70)
        time.sleep(0.5)
        self.servo_steer(80)
        time.sleep(0.5)
        self.servo_steer(90)
        time.sleep(0.5)
        self.servo_steer(100)
        time.sleep(1)
        self.servo_steer(0)

        print("Servo Check done.")


    def motor_check(self):
        print("Motor Check")
        #Testing Motors
        print("Testing motors...")

        self.motor_drive("A", 0)
        self.motor_drive("B", 0)

        print("  right fwd")
        self.motor_drive("A", 100)
        time.sleep(1)
        self.motor_drive("A", 0)
        time.sleep(0.5)

        print("  left fwd")
        self.motor_drive("B", 100)
        time.sleep(1)
        self.motor_drive("B", 0)
        time.sleep(0.5)

        print("  right back")
        self.motor_drive("A", 100)
        time.sleep(1)
        self.motor_drive("A", 0)
        time.sleep(0.5)

        print("  left back")
        self.motor_drive("B", 100)
        time.sleep(1)
        self.motor_drive("B", 0)
        time.sleep(0.5)


        print("  both fwd slow")
        # start fast
        self.motor_drive("A", 100)
        self.motor_drive("B", 100)
        time.sleep(0.1)
        self.motor_drive("A", 75)
        self.motor_drive("B", 75)
        time.sleep(2)
        self.motor_drive("A", 50)
        self.motor_drive("B", 50)
        time.sleep(2)
        self.motor_drive("A", 25)
        self.motor_drive("B", 25)
        time.sleep(2)
        self.motor_drive("A", 0)
        self.motor_drive("B", 0)
        time.sleep(0.5)


if __name__ == "__main__":

    led = machine.Pin(25, machine.Pin.OUT)
    picar = Car()
    
    distance = Distance()
    comm = Comm(picar, distance)
    keepalive_timer = AliveTimer(comm, 2000)
    keepalive_timer.start()
    distance_timer = DistanceTimer(comm, 100)
    distance_timer.start()


    time.sleep(1)
    #picar.perform_check(1)

    comm.process()
    




