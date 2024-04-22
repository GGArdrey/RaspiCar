import sys
import os

# Get the path to the directory containing the 'control' package
package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the package directory to the Python path
sys.path.append(package_dir)

from XboxInput import XboxInput
from LaneDetectionHough import LaneDetectionHough
from ControlManager import ControlManager
from CommandInterface import CommandInterface


xbox_input = XboxInput()
lane_detection_hough = LaneDetectionHough(enable_pov = False, camera_port=0)
command_interface = CommandInterface()

control_manager = ControlManager(command_interface, xbox_input, lane_detection_hough)

while True:
    control_manager.run()