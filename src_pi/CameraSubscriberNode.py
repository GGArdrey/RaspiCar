import logging

import cv2
import zmq
import time
from utils.message_utils import parse_image_message
from Node import Node


class CameraSubscriberNode(Node):
    def __init__(self, zmq_sub_url="tcp://raspberrypi.local:5555", log_level=logging.INFO):
        super().__init__(log_level=log_level)
        self.zmq_sub_url = zmq_sub_url
        self.zmq_context = zmq.Context()
        self.zmq_subscriber = self.zmq_context.socket(zmq.SUB)
        self.zmq_subscriber.connect(self.zmq_sub_url)
        self.zmq_subscriber.setsockopt(zmq.RCVHWM, 1)  # Set high water mark to 1 to drop old frames
        self.zmq_subscriber.setsockopt_string(zmq.SUBSCRIBE, 'camera')


    def start(self):
        while True:
            try:
                message = self.zmq_subscriber.recv_multipart()
                topic, image, timestamp = parse_image_message(message)
                dt = time.time() - timestamp
                self.log(f"Total time to receive image: {dt}", logging.DEBUG)
                if image is not None:
                    cv2.imshow('Received Camera Feed', image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                        break
                else:
                    self.log("Camera Subscriber Node: Error decoding frame.", logging.ERROR)
            except Exception as e:
                self.log(f"Error receiving frame: {e}", logging.ERROR)
                break

    def release(self):
        self.zmq_subscriber.close()
        self.zmq_context.term()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    subscriber_node = CameraSubscriberNode(log_level=logging.DEBUG)
    try:
        subscriber_node.start()
    except KeyboardInterrupt:
        pass
    finally:
        subscriber_node.release()
