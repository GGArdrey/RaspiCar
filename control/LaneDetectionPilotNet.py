import time

import cv2
from CarCommands import CarCommands
from IControlAlgorithm import IControlAlgorithm
import threading
from util import timer
from IObservable import IObservable
import tensorflow as tf
from PilotNet import PilotNet
from tensorflow.keras.models import load_model


class LaneDetectionPilotnet(IControlAlgorithm, IObservable):

    def __init__(self):
        IControlAlgorithm.__init__(self)
        IObservable.__init__(self)
        self.lock = threading.Lock()
        self.new_frame_condition = threading.Condition(self.lock)
        self.latest_frame = None
        self.car_commands = CarCommands()
        self.processing_thread = None
        self.running = False
        self.pilotnet = PilotNet("","")
        self.model = self.load_model()


    def read_inputs(self):
        with self.lock:
            return self.car_commands.copy()

    def load_model(self):
        #model = self.pilotnet.build_model()
        #return model.load_weights("/home/pi/models/cp-0091.ckpt")
        return load_model("/home/pi/models/cp-0091.ckpt")

    def start(self):
        if not self.running:
            print("Starting Lane Detection Hough Thread...")
            self.running = True
            self.processing_thread = threading.Thread(target=self.wait_and_process_frames)
            self.processing_thread.start()

    def stop(self):
        if self.running:
            self.running = False
            with self.new_frame_condition:
                self.new_frame_condition.notify()  # Wake up the thread if it's waiting
            if self.processing_thread:
                self.processing_thread.join()

    def update(self, frame, timestamp):
        with self.new_frame_condition:
            self.latest_frame = frame
            self.new_frame_condition.notify()  # Signal that a new frame is available

    def wait_and_process_frames(self):
        while True:
            with self.new_frame_condition:
                self.new_frame_condition.wait()  # Wait for a new frame
                frame = self.latest_frame
                if frame is None:
                    print("Received None Frame inside LaneDetectionTransform. Exit Thread...")
                    break  # Allow using None as a shutdown signal

            self.process_frame(frame)

    def process_frame(self, frame):
        with timer("LaneDetectionHough.process_frame Execution"):
            car_commands = CarCommands()
            frame = self.pilotnet.scale_and_crop_image(frame)

            input_tensor = tf.convert_to_tensor(frame)
            input_tensor = tf.expand_dims(input_tensor, 0)  # Create batch axis
            prediction = self.model.predict(input_tensor)[0][0]
            print("Prediction: ", prediction)
            car_commands.steer = float(prediction)


            with self.lock:
                self.car_commands = car_commands
            self._notify_observers(frame,timestamp = time.time())

    import cv2

    def scale_and_crop(self, frame, target_width=200, target_height=66):
        # Step 1: Determine the scaling factor
        # Calculate the ratio of the target dimensions to the current dimensions
        height, width = frame.shape[:2]
        scaling_factor = target_width / width

        # Step 2: Resize the image to maintain aspect ratio
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Step 3: Crop the resized image to the target dimensions
        # Calculate the starting y-coordinate of the crop (bottom part)
        if new_height > target_height:
            y_start = new_height - target_height
        else:
            y_start = 0

        # Perform the crop
        cropped_frame = resized_frame[y_start:y_start + target_height, 0:target_width]

        return cropped_frame


