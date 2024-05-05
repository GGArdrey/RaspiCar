# Master Makefile to handle the entire deployment process

# Define default target
all: upload-control upload-pico upload-to-pico

# Upload files from control/ directory to Raspberry Pi
upload-control:
	$(MAKE) -C control/ upload

# Upload files from pico/code/src/ directory to Raspberry Pi
upload-pico:
	$(MAKE) -C pico/ upload

# Upload files from Raspberry Pi to Pi Pico
# Use absolute path of rshell on the pi
upload-to-pico:
	ssh pi@raspberrypi.local 'cd /home/pi/pico/ && /home/pi/.local/bin/rshell cp src/* /pyboard/'

# List boards connected to Raspberry Pi
list-boards:
	ssh pi@raspberrypi.local '/home/pi/.local/bin/rshell boards'

# Open REPL session on Pi Pico through Raspberry Pi
open-shell:
	ssh pi@raspberrypi.local '/home/pi/.local/bin/rshell repl'

copy:
	scp -r pi@raspberrypi.local:/home/pi/data/* ./data/

upload-model:
	scp -r ./training/05-05-2024_10-21/checkpoints/cp-0024.tflite pi@raspberrypi.local:/home/pi/models/

copy-log:
	scp -r pi@raspberrypi.local:/home/pi/log.csv ./