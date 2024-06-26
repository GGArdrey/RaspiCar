import threading
import cv2
import os
from IObserver import IObserver
import numpy as np
from CarCommands import CarCommands
from datetime import datetime
class DataRecorder(IObserver):
    '''
    This class stores camera frames labled with current steering. It needs to observe an input and a camera. It
    stores pairs of a new frame and a new control input. Frames and Control inputs are received async, a timestamp is
    checked that indicate the creation of the frame and the creation of the steering commands. If they are too far
    apart, they are not stored
    '''
    def __init__(self,save_dir='/home/pi/data/'):
        super().__init__()
        self.latest_frame = None
        self.latest_frame_timestamp = None
        self.latest_commands = None
        self.latest_command_timestamp = None
        self.max_time_diff = 0.05
        self.lock = threading.Lock()
        self.new_data_event = threading.Event()  # Event to signal new data
        self.thread = None
        self.running = False
        self.save_dir = save_dir
        self.storage_dir = None
        self.image_count = 0


    def start(self):
        if not self.running:
            # start by creating directory
            date_time_dir = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            self.storage_dir = os.path.join(self.save_dir, date_time_dir)
            os.makedirs(self.storage_dir, exist_ok=True)  # Ensure the directory exists
            self.image_count = 0  # reset counter
            self.running = True
            self.thread = threading.Thread(target=self.run)
            self.thread.start()
            print("Data Recorder Thread started...")

    def stop(self):
        if self.running:
            self.running = False
            self.new_data_event.set()  # Unblock the thread if waiting
            self.thread.join()
            print("Data Recorder Thread stopped...")


    def run(self):
        while self.running:
            self.new_data_event.wait()
            if not self.running:
                break
            self.process_data()
            self.new_data_event.clear()

    def process_data(self):
        with self.lock:
            frame = self.latest_frame
            frame_time = self.latest_frame_timestamp
            commands = self.latest_commands
            command_time = self.latest_command_timestamp

        if frame is not None and commands is not None:
            if abs(frame_time - command_time) <= self.max_time_diff:
                self.save_frame_with_label(frame, commands)
                self.latest_frame = None  # Reset the frame to prevent resaving the same image
                self.latest_commands = None
            else:
                print("Dismiss Data, Frame and Control Command too async: ", abs(frame_time - command_time))


    def update(self, data, timestamp):
        with self.lock:
            if isinstance(data, np.ndarray):  # Frame data
                self.latest_frame = data
                self.latest_frame_timestamp = timestamp
            elif isinstance(data, CarCommands):  # Command data
                self.latest_commands = data
                self.latest_command_timestamp = timestamp
                if data.start_capture and not self.running:
                    self.start()
                    print("Started Recording...")
                if data.stop_capture and self.running:
                    print("Stopped Recording...")
                    self.stop()
            self.new_data_event.set()


    def save_frame_with_label(self, frame, commands):
        self.image_count += 1
        filename = f"{self.image_count}_{commands.steer:.2f}.jpg"
        filepath = os.path.join(self.storage_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"Saved {filename} with steering angle: {commands.steer:.2f}")
