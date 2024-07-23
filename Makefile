# Master Makefile to handle the entire deployment process

# Define default target
all: upload-code upload-to-pico

# Upload files from control/ directory to Raspberry Pi
upload-code:
	ssh pi@raspberrypi.local 'mkdir -p /home/pi/raspicar/src_pi/ && mkdir -p /home/pi/raspicar/src_pico/'
	scp -r ./src_pi pi@raspberrypi.local:/home/pi/raspicar/
	scp -r ./src_pico pi@raspberrypi.local:/home/pi/raspicar/

# Upload files from Raspberry Pi to Pi Pico
upload-to-pico:
	ssh pi@raspberrypi.local 'cd /home/pi/raspicar/src_pico/ && /home/pi/.local/bin/rshell cp * /pyboard/'

# List boards connected to Raspberry Pi
list-boards:
	ssh pi@raspberrypi.local '/home/pi/.local/bin/rshell boards'

# Open REPL session on Pi Pico through Raspberry Pi
open-shell:
	ssh pi@raspberrypi.local 'picocom -b 115200 /dev/ttyACM0'

copy:
	scp -r pi@raspberrypi.local:/home/pi/data/* ./data/

upload-model:
	scp -r ./training/05-05-2024_10-21/checkpoints/cp-0024.tflite pi@raspberrypi.local:/home/pi/models/

copy-log:
	scp -r pi@raspberrypi.local:/home/pi/log.csv ./