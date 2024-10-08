"""
RaspiCar
Copyright (c) 2024 Fynn Luca Maaß

Licensed under the Custom License. See the LICENSE file in the project root for license terms.
"""


import multiprocessing
import argparse
import logging
from CameraNode import CameraNode
from XboxGamepadNode import XboxGamepadNode
from Node import Node
from DataRecorderNode import DataRecorderNode
from GamepadCommandNode import GamepadCommandNode
from ControlFusionNode import ControlFusionNode
from UARTInterfaceNode import UARTInterfaceNode
from PilotNetCNode import PilotNetCNode
from LaneDetectionNode import LaneDetectionNode
from YourAlgorithmTemplateNode import YourAlgorithmTemplateNode
import time
import signal
import os

# Function to map string log level to logging level
def get_log_level(log_level_str):
    log_level_str = log_level_str.upper()
    if log_level_str == 'DEBUG':
        return logging.DEBUG
    elif log_level_str == 'INFO':
        return logging.INFO
    elif log_level_str == 'WARNING':
        return logging.WARNING
    elif log_level_str == 'ERROR':
        return logging.ERROR
    elif log_level_str == 'CRITICAL':
        return logging.CRITICAL
    else:
        raise ValueError(f"Invalid log level: {log_level_str}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Start nodes with specific logging levels.')
    parser.add_argument('--log', type=str, default='INFO', help='Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    args = parser.parse_args()

    # Get the logging level
    log_level = get_log_level(args.log)

    # Here you define all nodes to start with their parameters. The default only specifies a log-level for every node.
    # The args can be expanded to also change their other default parameters. You need to specify them in the correct order
    # i.e. {"node_class": CameraNode, "args": (log_level, 10, 0, 640, 360,)}
    # to also specify Framerate, CameraID, Width, Height
    # Note: In the args list, you need a comma after the last parameter if you not exhaustivly specified all parameters the node takes
    node_configs = [
        {"node_class": ControlFusionNode, "args": (log_level,)},
        {"node_class": UARTInterfaceNode, "args": ('INFO',)},
        {"node_class": CameraNode, "args": (log_level,)},
        {"node_class": XboxGamepadNode, "args": (log_level,)},
        {"node_class": GamepadCommandNode, "args": (log_level,)},
        {"node_class": DataRecorderNode, "args": (log_level,0.15,'/home/pi/data/','tcp://localhost:5550','camera', 'tcp://localhost:5541','gamepad_function_commands','tcp://localhost:5570','fused_steering_commands')},
        {"node_class": PilotNetCNode, "args": (log_level,)}, # HERE YOU CAN CHANGE YOUR SELF IMPLEMENTED NODE
    ]
    max_time_diff = 0.1,
    save_dir = '/home/pi/data/',
    image_sub_url = "tcp://localhost:5550",
    image_sub_topic = "camera",
    gamepad_function_sub_url = "tcp://localhost:5541",
    gamepad_function_sub_topic = "gamepad_function_commands",
    steering_commands_url = "tcp://localhost:5541",
    steering_commands_topic = "gamepad_steering_commands",
    pico_data_url = "tcp://localhost:5580",
    pico_data_topic = "pico_data"

    processes = []

    # Execute all nodes from above with their respective parameters specified
    for config in node_configs:
        p = multiprocessing.Process(target=Node.start_node_process, args=(config["node_class"], *config["args"]))
        processes.append(p)
        p.start()
        print(f"Started {config['node_class'].__name__} with PID {p.pid}")


    def terminate_all_processes(process_list):
        '''
        Kill all processes running
        '''
        for proc in process_list:
            if proc.is_alive():
                print(f"Terminating process {proc.pid}")
                proc.terminate()
                proc.join(timeout=0.3)
                if proc.is_alive():
                    print(f"Force killing process {proc.pid}")
                    os.kill(proc.pid, signal.SIGKILL)

    # Watch all processes (nodes). If one terminates, terminate the whole system.
    try:
        while True:
            crashed = False
            for p in processes:
                if not p.is_alive():
                    print(f"Process {p.pid} crashed. Terminating all processes...")
                    crashed = True
                    break
            if crashed:
                terminate_all_processes(processes)
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Terminating all processes...")
        terminate_all_processes(processes)

    print("All processes terminated.")
