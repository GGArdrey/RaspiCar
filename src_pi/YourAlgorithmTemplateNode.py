"""
RaspiCar
Copyright (c) 2024 (Fynn Luca Maaß) &

Licensed under the Custom License. See the LICENSE file in the project root for license terms.
"""

from utils.timer_utils import timer
import zmq
from utils.message_utils import create_json_message, parse_jpg_image_message
import logging
from Node import Node


class YourAlgorithmTemplateNode(Node):

    def __init__(self, log_level=logging.INFO,
                 zmq_pub_url="tcp://localhost:5560",
                 zmq_pub_topic="steering_commands",
                 camera_sub_url="tcp://localhost:5550",
                 camera_sub_topic="camera"):
        super().__init__(log_level=log_level)

        # ZeroMQ setup
        self.zmq_pub_url = zmq_pub_url
        self.zmq_pub_topic = zmq_pub_topic
        self.camera_sub_url = camera_sub_url
        self.camera_sub_topic = camera_sub_topic

        self.zmq_context = zmq.Context()
        self.zmq_publisher = self.zmq_context.socket(zmq.PUB)
        self.zmq_publisher.bind(self.zmq_pub_url)

        self.zmq_subscriber = self.zmq_context.socket(zmq.SUB)
        self.zmq_subscriber.connect(self.camera_sub_url)
        self.zmq_subscriber.setsockopt(zmq.CONFLATE, 1)  # Keep only the latest message
        self.zmq_subscriber.setsockopt(zmq.RCVHWM, 1)  # Set high water mark to 1 to drop old frames

        self.zmq_subscriber.setsockopt_string(zmq.SUBSCRIBE, self.camera_sub_topic)

    def start(self):
        while True:
            message = self.zmq_subscriber.recv_multipart()
            topic, image, timestamp = parse_jpg_image_message(message)
            # Execute your algorithm here
            self.predict(image, timestamp)

    def release(self):
        self.zmq_publisher.close()
        self.zmq_subscriber.close()
        self.zmq_context.term()

    def predict(self, frame, timestamp):
        '''
        This is where you should put your algorithm to predict steering commands.
        You can use the frame to predict steering commands.
        :param frame: normal openCV frame
        :param timestamp: timestamp of the frame when it was created
        :return:
        '''

        with timer() as get_elapsed_time:
            # If you do want to measure inference, put your code here
            # You can then measure execution times or inference times
            # something like calculated_steering_value = model.predict(frame)
            calculated_steering_value = 0.0

        # log elapsed time
        elapsed_time = get_elapsed_time()
        self.log(f"Elapsed Time: {elapsed_time}", logging.DEBUG)

        # BTW, here are all the logging levels you can use:
        # you can change logging levels to see more or less information when instanciaing the class
        # logging.DEBUG
        # logging.INFO
        # logging.WARNING
        # logging.ERROR
        # logging.CRITICAL

        # After predicting or calculating steering commands, you can publish them.
        # They must be subscribed by ControlFusionNode, which they are by default when you use the template.
        # You must publish a dictionary with the following keys and you can change values as you wish:
        payload = {
            "steer": calculated_steering_value,
            "throttle": 0,
            "emergency_stop": 0,
            "reset_emergency_stop": 0,
            "sensors_enable": 0,
            "sensors_disable": 0
        }

        # Publish the message
        # You can take a new time.time() timestamp or use the one you received from the camera depending on what you need
        # I would recommend using the one you received from the camera to measure latency end-to-end
        # The ControlFusionNode will measure latencies and gives warnings when its too high. THis would mean
        # that your algorithm is too slow or transmitting messages like large images is taking too long. High latency is
        # not catastrophic, it just means that the steering in not very responsive
        message = create_json_message(payload, self.zmq_pub_topic, timestamp=timestamp)
        self.zmq_publisher.send(message)


if __name__ == "__main__":
    # typically you dont execute this node as standalone. It is executed by the main.py script with the other nodes,
    # where you would need to add it
    a = YourAlgorithmTemplateNode()
    a.start()
