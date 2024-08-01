import multiprocessing
import argparse
import logging
from CameraNode import CameraNode
from XboxGamepadNode import XboxGamepadNode
from Node import Node
from DataRecorderNode import DataRecorderNode
from GamepadCommandNode import GamepadCommandNode
from ControlFusionNode import ControlFusionNode
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

    node_configs = [
        {"node_class": ControlFusionNode, "args": (log_level,)},
        {"node_class": CameraNode, "args": (log_level,)},
        {"node_class": XboxGamepadNode, "args": (log_level,)},
        {"node_class": GamepadCommandNode, "args": (log_level,)},
        {"node_class": DataRecorderNode, "args": (log_level,)},
        {"node_class": PilotNetCNode, "args": (log_level,)},
    ]


    processes = []

    for config in node_configs:
        p = multiprocessing.Process(target=Node.start_node_process, args=(config["node_class"], *config["args"]))
        processes.append(p)
        p.start()
        print(f"Started {config['node_class'].__name__} with PID {p.pid}")


    def terminate_all_processes(process_list):
        for proc in process_list:
            if proc.is_alive():
                print(f"Terminating process {proc.pid}")
                proc.terminate()
                proc.join(timeout=0.3)
                if proc.is_alive():
                    print(f"Force killing process {proc.pid}")
                    os.kill(proc.pid, signal.SIGKILL)


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
