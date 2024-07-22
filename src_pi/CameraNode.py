from utils.message_utils import create_image_message
import cv2
import zmq
from Node import Node
import time
import logging

class CameraNode(Node):
    def __init__(self, frame_rate = 10, camera_source=0, frame_width=None, frame_height=None, zmq_pub_url="tcp://*:5555", pub_topic='camera', log_level=logging.INFO):
        super().__init__(log_level=log_level)
        self.camera_source = camera_source
        self.frame_rate = frame_rate
        self.prev_time = 0
        self.cap = cv2.VideoCapture(camera_source)
        if frame_width and frame_height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)


        self.zmq_pub_url = zmq_pub_url
        self.pub_topic = pub_topic
        self.zmq_context = zmq.Context()
        self.zmq_publisher = self.zmq_context.socket(zmq.PUB)
        self.zmq_publisher.bind(self.zmq_pub_url)


    def start(self):
        while True:
            time_elapsed = time.time() - self.prev_time
            ret, frame = self.cap.read()
            if ret:
                if time_elapsed > 1./self.frame_rate:
                    self.prev_time = time.time()
                    message = create_image_message(frame, self.pub_topic)
                    self.zmq_publisher.send_multipart(message)
            else:
                self.log("Error capturing frame.", logging.ERROR)
                break


    def release(self):
        if self.cap.isOpened():
            self.cap.release()
            self.zmq_publisher.close()
            self.zmq_context.term()
