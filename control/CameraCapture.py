import time

import cv2
import threading
from IObservable import IObservable

class CameraCapture(IObservable):
    def __init__(self, camera_source=0, frame_width=None, frame_height=None):
        super().__init__()
        self.camera_source = camera_source
        self.cap = cv2.VideoCapture(camera_source)
        if frame_width and frame_height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.frame_lock = threading.Lock()
        self.current_frame = None
        self.running = False
        self.capture_thread = None

    def start(self):
        if not self.running:
            print("Starting Camera Capture Thread..")
            self.running = True
            self.capture_thread = threading.Thread(target=self.update_frame)
            self.capture_thread.start()

    def stop(self):
        if self.running:
            self.running = False
            if self.capture_thread:
                self.capture_thread.join()
            self.release()

    def update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.current_frame = frame
                self._notify_observers(frame.copy(), timestamp = time.time()) #TODO copy()?
            else:
                print("Error capturing frame.")
                break

    def get_frame(self):
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def release(self):
        with self.frame_lock:
            if self.cap.isOpened():
                self.cap.release()
                print("Camera released.")