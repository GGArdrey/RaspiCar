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

xbox_input = XboxInput()
xbox_input.start()  # Start the input polling thread

lane_detector = LaneDetectionHough()
lane_detector.start()  # Begin processing frames thread

server = RPiServer()
server.start()  # starting acceting and sending thread inside server

camera = CameraCapture()
camera.register_observer(server)
camera.register_observer(lane_detector)
camera.start()  # Start capturing frames and sending it to observers thread

command_interface = CommandInterface()
control_manager = ControlManager(command_interface, xbox_input, lane_detector)
control_manager.start() # start thread
