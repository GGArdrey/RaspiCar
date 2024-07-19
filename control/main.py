
import sys
import os
# Get the path to the directory containing the 'control' package
package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the package directory to the Python path
sys.path.append(package_dir)

os.environ['SDL_AUDIODRIVER'] = 'dsp'


def record():
    from XboxInput import XboxInput
    from RPiServer import RPiServer
    from CameraCapture import CameraCapture
    from DataRecorder import DataRecorder
    from ControlManager import ControlManager
    from CommandInterface import CommandInterface


    recorder = DataRecorder()
    xbox_input = XboxInput(input_freq=5)
    xbox_input.register_observer(recorder)
    xbox_input.start()  # Start the input polling thread

    # Server for sending Video over the network
    server = RPiServer()
    server.start()

    camera = CameraCapture(frame_width=640, frame_height=360) #frame_width=640, frame_height=360
    camera.register_observer(recorder) #recorder want to get images
    camera.register_observer(server) #server wants to get images
    camera.start()  # Start camera thread


    command_interface = CommandInterface() # sends steering commands to src_pico
    control_manager = ControlManager(command_interface, xbox_input, control_algorithm1 = None, control_algorithm2 = None)
    xbox_input.register_observer(control_manager) #control manager wants to get inputs from controller
    control_manager.start()  # start thread


def drivePilotNet():
    from XboxInput import XboxInput
    from LaneDetectionPilotNet import LaneDetectionPilotnet
    from RPiServer import RPiServer
    from CameraCapture import CameraCapture
    from ControlManager import ControlManager
    from CommandInterface import CommandInterface
    xbox_input = XboxInput()
    xbox_input.start()  # Start the input polling thread

    server = RPiServer()

    lane_detector = LaneDetectionPilotnet()
    lane_detector.register_observer(server)
    lane_detector.start()  # Begin processing frames thread

    server.start()  # starting acceting and sending thread inside server

    camera = CameraCapture(frame_width=640, frame_height=360)
    #camera.register_observer(server)
    camera.register_observer(lane_detector)
    camera.start()  # Start capturing frames and sending it to observers thread

    command_interface = CommandInterface()
    control_manager = ControlManager(command_interface, xbox_input, lane_detector, None)
    lane_detector.register_observer(control_manager)
    control_manager.start()  # start thread


def driveLaneDetection():
    from XboxInput import XboxInput
    from RPiServer import RPiServer
    from CameraCapture import CameraCapture
    from LaneDetectionPolyFit import LaneDetectionPolyfit
    from ControlManager import ControlManager
    from CommandInterface import CommandInterface
    xbox_input = XboxInput()
    xbox_input.start()  # Start the input polling thread

    server = RPiServer()
    server.start()  # starting acceting and sending thread inside server


    #lane_detector = LaneDetectionHough()
    lane_detector = LaneDetectionPolyfit()
    lane_detector.register_observer(server)
    lane_detector.start()  # Begin processing frames thread

    camera = CameraCapture(frame_width=640, frame_height=360)
    #camera.register_observer(server)
    camera.register_observer(lane_detector)
    camera.start()  # Start capturing frames and sending it to observers thread

    command_interface = CommandInterface()
    control_manager = ControlManager(command_interface, xbox_input, lane_detector, None)
    lane_detector.register_observer(control_manager)
    control_manager.start()  # start thread

if __name__ == "__main__":
    record()
    #drivePilotNet()
    #driveLaneDetection()