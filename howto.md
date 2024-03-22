# How To: **Picar** 

	Make sure raspicar and your laptop that you are accessing it from are in the same network.

## WIFI in office to connect to picar:


## Raspberry Pi (High level Control Unit):
    
### Connecting:
    
- ssh pi@raspberrypi

    or

- ssh pi@raspberrypi.local

---
### Controlling the car via "WASD" controls:
        
in home directory use:
        
    ./control/control.py

| Key | Function |
| --- | --- |
| w | forward |
| s | backward |
| a | servo left (50%) |
| d | servo right (50%) |
| SPACEBAR | stop motors, center servo |
| ! | toggles sensors |

leave this mode with ctrl + c

### Controlling car via commands: 

    picocom -b 115200 /dev/ttyS0

#### Commands:

| Commands | Function |
| ----------- | ----------- |
| "drive,x" | x from -100 to +100 where negative numbers are backwards and positive numbers are forward, "drive,0" == stop |
| "steer,x" | x from -100 to +100 where negative numbers are left and positive numbers are right, "steer,0" == center |
|"toggleSensors" | Toggles in betweeen permanent servo output and just when min_dist is hit output |
| "check" | performs a check of the whole motor and servo setup  |
| "servo_check" | performs a check of the steering servo |
| "motor_check" | performs a check of the driving motors |
| "toggleKA" | toggles the keep alive checks (stopping the car if the controllers time out) |
| "drivef" | slow forward drive |
| "driveb" | slow backward drive |
| "stop" | stop -> stops the motors and centers servo |

leave the shell via ctrl + x

### Wrapple "shell on the pico" (debug):
        
    picocom -b 115200 /dev/ttyACM0

after shell opens use ctrl + d to view live debug

leave the shell with crt + a then crtl + x

### Flashing new code onto pico / raspi:

Code for pico:
    
    use makefile in the pico folder

For control code (raspi):
    
    use makefile in cmdline folder

