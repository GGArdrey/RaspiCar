from utils.message_utils import create_image_message, create_jpg_image_message
import cv2
import zmq
from Node import Node
import time
import logging
from threading import Thread, Lock


class CameraNode(Node):
    def __init__(self, frame_rate=10, camera_source=0, frame_width=None, frame_height=None, zmq_pub_url="tcp://*:5555",
                 pub_topic='camera', log_level=logging.INFO):
        super().__init__(log_level=log_level)
        self.camera_source = camera_source
        self.frame_duration = 1 / frame_rate
        self.frame_rate = frame_rate
        self.cap = cv2.VideoCapture(camera_source)
        if frame_width and frame_height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        self.zmq_pub_url = zmq_pub_url
        self.pub_topic = pub_topic
        self.pub_topic_compressed = pub_topic + "_compressed"
        self.zmq_context = zmq.Context()
        self.zmq_publisher = self.zmq_context.socket(zmq.PUB)
        self.zmq_publisher.setsockopt(zmq.SNDHWM, 1)  # Set high water mark to 1 to drop old frames
        self.zmq_publisher.bind(self.zmq_pub_url)

        self.latest_frame = None
        self.frame_lock = Lock()
        self.running = False

    def start(self):
        self.running = True
        capture_thread = Thread(target=self.capture_frames)
        capture_thread.start()

        while self.running:
            start_time = time.time()

            with self.frame_lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()
                    message = create_jpg_image_message(frame, self.pub_topic, quality=95)
                    try:
                        self.zmq_publisher.send_multipart(message, flags=zmq.NOBLOCK)
                    except zmq.Again:
                        self.log("Publisher queue is full, dropping frame.", logging.WARNING)

            # Calculate elapsed time and sleep to maintain the desired frame rate
            elapsed_time = time.time() - start_time
            sleep_time = max(0, self.frame_duration - elapsed_time)
            time.sleep(sleep_time)

    def capture_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
            else:
                self.log("Error capturing frame.", logging.ERROR)
                self.running = False

    def release(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.zmq_publisher.close()
        self.zmq_context.term()