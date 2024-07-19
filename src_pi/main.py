import multiprocessing
from CameraNode import CameraNode
from XboxGamepadNode import XboxGamepadNode
from Node import Node
from DataRecorderNode import DataRecorderNode
from XboxGamepadTranslatorNode import XboxGamepadTranslatorNode
from ControlFusionNode import ControlFusionNode
from PilotNetCNode import PilotNetCNode

if __name__ == "__main__":
    # Example configuration for multiple nodes
    node_configs = [
        {"node_class": ControlFusionNode, "args": ("tcp://localhost:5560", "pilotnetc_steering_commands",
                                                   "tcp://localhost:5557", "gamepad_steering_commands",
                                                   "tcp://localhost:5570", "fused_steering_commands", 1)},
        {"node_class": CameraNode, "args": (0, 640, 360, "tcp://*:5555", 'camera')},
        {"node_class": XboxGamepadNode, "args": (0, 10, "tcp://localhost:5556", "gamepad")},
        {"node_class": XboxGamepadTranslatorNode, "args": ("tcp://localhost:5556", "gamepad", "tcp://localhost:5557")},
        {"node_class": DataRecorderNode, "args": ("tcp://localhost:5555", "camera",
                                                  "tcp://localhost:5557", "gamepad_function_commands", "gamepad_steering_commands",
                                                  "./data/")},
        {"node_class": PilotNetCNode, "args": ("tcp://localhost:5560", "pilotnetc_steering_commands", "tcp://localhost:5555", "camera")}
    ]

    processes = []

    for config in node_configs:
        p = multiprocessing.Process(target=Node.start_node_process, args=(config["node_class"], *config["args"]))
        processes.append(p)
        p.start()
        print(f"Started {config['node_class'].__name__} with PID {p.pid}")

    for p in processes:
        p.join()
