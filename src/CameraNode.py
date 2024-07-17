import time
import cv2
import zmq
from Node import Node


class CameraNode(Node):
    def __init__(self, camera_source=0, frame_width=None, frame_height=None, zmq_pub_url="tcp://*:5555"):
        super().__init__()
        self.camera_source = camera_source
        self.cap = cv2.VideoCapture(camera_source)
        if frame_width and frame_height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.zmq_pub_url = zmq_pub_url
        self.zmq_context = zmq.Context()
        self.zmq_publisher = self.zmq_context.socket(zmq.PUB)
        self.zmq_publisher.bind(self.zmq_pub_url)

    def start(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                _, buffer = cv2.imencode('.jpg', frame)  # Encoding frame to JPEG format
                timestamp = time.time()
                self.zmq_publisher.send_multipart([b'frame', buffer.tobytes(), str(timestamp).encode('utf-8')])
            else:
                print("Camera Node: Error capturing frame.")
                break

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
            self.zmq_publisher.close()
            self.zmq_context.term()
            print("Camera Node released and ZeroMQ publisher closed.")
