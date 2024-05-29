import time

import cv2
from CarCommands import CarCommands
from IControlAlgorithm import IControlAlgorithm
import threading
from util import timer
from IObservable import IObservable
import tensorflow as tf
from PilotNet import PilotNet
import numpy as np
from tflite_runtime.interpreter import Interpreter


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
        #self.model = self.load_model()

        self.interpreter = Interpreter(model_path="/home/pi/models/cp-0024.tflite")
        self.interpreter.allocate_tensors()
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


    def read_inputs(self):
        with self.lock:
            return self.car_commands.copy()

    def load_model(self):
        #model = self.pilotnet.build_model()
        #return model.load_weights("/home/pi/models/cp-0091.ckpt")
        #return tf.keras.models.load_model('/home/pi/models/cp-0001.keras', custom_objects={'StandardizationLayer': StandardizationLayer})
        return tf.keras.models.load_model('/home/pi/models/model.keras')

    def load_model_tflite(self):
        raise NotImplementedError

    def start(self):
        if not self.running:
            print("Starting Lane Detection PilotNet Thread...")
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

            self.process_frame_coral_classification(frame)

    def process_frame(self, frame):
        car_commands = CarCommands()
        #frame = self.pilotnet.bgr_to_hsv(frame)
        frame = self.pilotnet.resize_and_crop_image(frame)

        input_tensor = tf.convert_to_tensor(frame)
        input_tensor = tf.expand_dims(input_tensor, 0)  # Create batch axis
        with timer("Inference Time"):
            prediction = self.model.predict(input_tensor)[0][0]
        print("Prediction: ", prediction)
        car_commands.steer = float(prediction)


        with self.lock:
            self.car_commands = car_commands
        self._notify_observers(frame,timestamp = time.time())

    def process_frame_coral(self, frame):
        car_commands = CarCommands()

        # Assuming resize_and_crop_image function processes the frame as needed for the model
        frame = self.pilotnet.resize_and_crop_image(frame)

        # Prepare the frame for the model (make sure the datatype matches what the model expects)
        frame = np.array(frame, dtype=np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], [frame])

        with timer("Inference Time"):
            self.interpreter.invoke()
            prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]

        print("Prediction: ", prediction)
        car_commands.steer = float(prediction)

        with self.lock:
            self.car_commands = car_commands
        self._notify_observers(frame, timestamp=time.time())


    def process_frame_coral_classification(self, frame):
        classes = [-1.0,  -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6,  0.8, 1.0]
        car_commands = CarCommands()

        # Assuming resize_and_crop_image function processes the frame as needed for the model
        frame = self.pilotnet.resize_and_crop_image(frame)

        # Prepare the frame for the model (make sure the datatype matches what the model expects)
        frame = np.array(frame, dtype=np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], [frame])

        with timer("Inference Time"):
            self.interpreter.invoke()
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        # Use argmax to find the most likely class label
        predicted_class = np.argmax(predictions)

        # Calculate the neighborhood probabilities
        max_prob = predictions[predicted_class]
        left_prob = predictions[predicted_class - 1] if predicted_class > 0 else 0
        right_prob = predictions[predicted_class + 1] if predicted_class < len(predictions) - 1 else 0
        neighborhood_prob = max_prob + left_prob + right_prob


        # Calculate the probabilities and angles for steering calculation
        total_prob = max_prob + left_prob + right_prob
        weighted_steer_value = (left_prob * classes[predicted_class - 1] if predicted_class > 0 else 0) + \
                      (max_prob * classes[predicted_class]) + \
                      (right_prob * classes[predicted_class + 1] if predicted_class < len(predictions) - 1 else 0)

        # Normalize steer value if total_prob is not zero
        weighted_steer_value = weighted_steer_value / total_prob if total_prob > 0 else 0


        print("Predictions:")
        for i, pred in enumerate(predictions):
            print(f"  {classes[i]}: {pred * 100:.2f}%")

        print(f"Predicted Class: {classes[predicted_class]}")

        # Print the neighborhood probability
        print(f"Predicted Class Probability: {max_prob * 100:.2f}%")
        print(f"Predicted Class 1-Neighborhood: {neighborhood_prob * 100:.2f}%")
        print(f"Average Prediction in 1-Neighborhood: {weighted_steer_value:.2f}")
        current_time = round(time.time() * 1000)
        car_commands.steer = weighted_steer_value  # Assuming steer is now used to represent the predicted class
        car_commands.additional_info = [current_time, classes[predicted_class],weighted_steer_value,max_prob,neighborhood_prob,predictions]
        with self.lock:
            self.car_commands = car_commands
        self._notify_observers(frame, timestamp=time.time())
