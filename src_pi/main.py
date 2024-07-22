import multiprocessing
import argparse
import logging
from CameraNode import CameraNode
from XboxGamepadNode import XboxGamepadNode
from Node import Node
from DataRecorderNode import DataRecorderNode
from XboxGamepadTranslatorNode import XboxGamepadTranslatorNode
from ControlFusionNode import ControlFusionNode
from PilotNetCNode import PilotNetCNode

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

    # Example configuration for multiple nodes
    node_configs = [
        {"node_class": ControlFusionNode, "args": ("tcp://localhost:5560", "pilotnetc_steering_commands",
                                                  "tcp://localhost:5557", "gamepad_steering_commands",
                                                  "tcp://localhost:5570", "fused_steering_commands", 0.0, log_level)},
        {"node_class": CameraNode, "args": (10, 0, 640, 360, "tcp://*:5555", 'camera', log_level)},
        {"node_class": XboxGamepadNode, "args": (0, 20, "tcp://localhost:5556", "gamepad", log_level)},
        {"node_class": XboxGamepadTranslatorNode, "args": ("tcp://localhost:5556", "gamepad", "tcp://localhost:5557", log_level)},
        {"node_class": DataRecorderNode, "args": ("tcp://localhost:5555", "camera",
                                                  "tcp://localhost:5557", "gamepad_function_commands", "gamepad_steering_commands",
                                                  "./data/", log_level)},
        {"node_class": PilotNetCNode, "args": ("tcp://localhost:5560", "pilotnetc_steering_commands", "tcp://localhost:5555", "camera", log_level)}
    ]

    processes = []

    for config in node_configs:
        p = multiprocessing.Process(target=Node.start_node_process, args=(config["node_class"], *config["args"]))
        processes.append(p)
        p.start()
        print(f"Started {config['node_class'].__name__} with PID {p.pid}")

    for p in processes:
        p.join()
