import sys
import os
import threading

# Get the path to the directory containing the 'control' package
package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the package directory to the Python path
sys.path.append(package_dir)

from XboxInput import XboxInput
from LaneDetectionHough import LaneDetectionHough
from ControlManager import ControlManager
from CommandInterface import CommandInterface
from RPiServer import RPiServer
from CameraCapture import CameraCapture


camera_capture = CameraCapture()
rpi_server = RPiServer()
lane_detection_hough = LaneDetectionHough()

camera_capture.register_observer(rpi_server)
camera_capture.register_observer(lane_detection_hough)


xbox_input = XboxInput()
command_interface = CommandInterface()
control_manager = ControlManager(command_interface, xbox_input, None)



capture_thread = threading.Thread(target=camera_capture.update_frame)
#capture_thread.daemon = True  # Set as daemon thread to stop when main program exits
capture_thread.start()
print("Started Camera Capture...")

server_thread = threading.Thread(target=rpi_server.start_accepting_connections)
#server_thread.daemon = True  # Set as daemon thread to stop when main program exits
server_thread.start()
print("Started Server...")

control = threading.Thread(target=control_manager.run)
#control.daemon = True  # Set as daemon thread to stop when main program exits
control.start()
print("Started Control Manager...")







while True:
    control_manager.run()