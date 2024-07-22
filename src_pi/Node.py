import logging


class Node:
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)  # Default logging level, can be changed

        # Create a console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)


    def log(self, message, level=logging.INFO):
        self.logger.log(level, message)


    def start(self):
        # Abstract method to start the node's main functionality.
        # Subclasses must override this method to provide specific implementation.
        raise NotImplementedError("Subclasses should implement this method.")

    def release(self):
        # Abstract method to release resources used by the node.
        # Subclasses must override this method to provide specific implementation.
        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    def start_node_process(node_class, *args):
        # Static method to initialize and start a node process.
        # This method is used to create and run an instance of the node class.
        node_instance = node_class(*args)  # Create an instance of the specified node class with given arguments
        try:
            node_instance.start()  # Start the node's main functionality
            node_instance.log(f"{node_class.__name__} started.", logging.INFO)
        except Exception as e:
            node_instance.log(f"{e}.", logging.ERROR)
        finally:
            node_instance.release()  # Ensure resources are released when done or if an error occurs
            node_instance.log(f"{node_class.__name__} released.", logging.INFO)
