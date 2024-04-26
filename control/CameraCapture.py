import cv2
import threading
from IObservable import Observable


class CameraCapture(Observable):
    def __init__(self, camera_source=0):  # TODO height width of camera frame
        super().__init__()
        self.camera_source = camera_source
        self.cap = cv2.VideoCapture(camera_source) # TODO: set framerate
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_WIDTH)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_HEIGHT)
        self.frame_lock = threading.Lock()
        self.current_frame = None

    def update_frame(self):
        '''
        Notifies all its observers and provides a frame. Additionally, the current frame is also saved in this class,
        protected by a mutex
        :return:
        '''
        while True:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.current_frame = frame
                self.notify_observers(frame)
            else:
                print("Error capturing frame.")

    def get_frame(self):
        with self.frame_lock:
            return self.current_frame

    def release(self):
        self.cap.release()
