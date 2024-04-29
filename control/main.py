import sys
import os


# Get the path to the directory containing the 'control' package
package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the package directory to the Python path
sys.path.append(package_dir)

from XboxInput import XboxInput
from LaneDetectionHough import LaneDetectionHough
from LaneDetectionPilotNet import LaneDetectionPilotnet
from ControlManager import ControlManager
from CommandInterface import CommandInterface
from RPiServer import RPiServer
from CameraCapture import CameraCapture
from LaneDetectionPolyFit import LaneDetectionPolyfit
from DataRecorder import DataRecorder


def capture():
    recorder = DataRecorder()

    xbox_input = XboxInput(input_freq=30)
    xbox_input.register_observer(recorder)
    xbox_input.start()  # Start the input polling thread

    camera = CameraCapture(frame_width=640, frame_height=360)
    camera.register_observer(recorder)
    camera.start()  # Start capturing frames and sending it to observers thread

    command_interface = CommandInterface()
    control_manager = ControlManager(command_interface, xbox_input, None)
    control_manager.start()  # start thread

    recorder.start()

def drive():
    xbox_input = XboxInput()
    xbox_input.start()  # Start the input polling thread

    #server = RPiServer()
    #server.start()  # starting acceting and sending thread inside server

    # lane_detector = LaneDetectionHough()
    lane_detector = LaneDetectionPilotnet()
    # lane_detector.register_observer(server)
    # lane_detector.start()  # Begin processing frames thread

    camera = CameraCapture(frame_width=640, frame_height=360)
    #camera.register_observer(server)
    camera.register_observer(lane_detector)
    camera.start()  # Start capturing frames and sending it to observers thread

    command_interface = CommandInterface()
    control_manager = ControlManager(command_interface, xbox_input, lane_detector)
    control_manager.start()  # start thread

if __name__ == "__main__":
    drive()