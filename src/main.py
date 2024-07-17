import multiprocessing
from CameraNode import CameraNode
from XboxGamepadNode import XboxGamepadNode
from Node import Node
from DataRecorderNode import DataRecorderNode

if __name__ == "__main__":
    # Example configuration for multiple nodes
    node_configs = [
        {"node_class": CameraNode, "args": (0, 640, 480, "tcp://*:5555")},
        {"node_class": XboxGamepadNode, "args": (1, 10, "tcp://*:5556")},
        {"node_class": DataRecorderNode, "args": ("tcp://localhost:5555", "tcp://localhost:5556", "./data/")}
    ]

    processes = []

    for config in node_configs:
        p = multiprocessing.Process(target=Node.start_node_process, args=(config["node_class"], *config["args"]))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
