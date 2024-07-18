import multiprocessing
from CameraNode import CameraNode
from XboxGamepadNode import XboxGamepadNode
from Node import Node
from DataRecorderNode import DataRecorderNode
from XboxGamepadTranslatorNode import XboxGamepadTranslatorNode

if __name__ == "__main__":
    # Example configuration for multiple nodes
    node_configs = [
        {"node_class": CameraNode, "args": (2, 640, 360, "tcp://*:5555", 'camera')},
        {"node_class": XboxGamepadNode, "args": (0, 20, "tcp://*:5556", "gamepad")},
        {"node_class": XboxGamepadTranslatorNode, "args": ("tcp://localhost:5556", "gamepad", "tcp://*:5557")},
        {"node_class": DataRecorderNode, "args": ("tcp://localhost:5555", "camera",
                                                  "tcp://localhost:5557", "gamepad_function_commands", "gamepad_steering_commands",
                                                  "./data/")}
    ]

    processes = []

    for config in node_configs:
        p = multiprocessing.Process(target=Node.start_node_process, args=(config["node_class"], *config["args"]))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
