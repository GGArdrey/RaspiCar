import time

import cv2
import numpy as np
from CarCommands import CarCommands
from IControlAlgorithm import IControlAlgorithm
import threading
from util import timer
from IObservable import IObservable


class LaneDetectionPolyfit(IControlAlgorithm, IObservable):

    def __init__(self):
        IControlAlgorithm.__init__(self)
        IObservable.__init__(self)
        self.lock = threading.Lock()
        self.new_frame_condition = threading.Condition(self.lock)
        self.latest_frame = None
        self.car_commands = CarCommands()
        self.processing_thread = None
        self.running = False

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
            FRAME_WIDTH, FRAME_HEIGHT, _ = frame.shape
            car_commands = CarCommands()

            img = self.denoise_frame(frame)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = self.detect_edges(img)
            FRAME_WIDTH, FRAME_HEIGHT = img.shape
            img = self.mask_region(img,FRAME_HEIGHT, FRAME_WIDTH)

            FRAME_WIDTH, FRAME_HEIGHT = img.shape
            left_lines, right_lines, road_center = self.detect_curves(img, FRAME_HEIGHT, FRAME_WIDTH)

            # Compute steering based on road center
            FRAME_WIDTH, FRAME_HEIGHT = img.shape
            print("center: ", road_center)
            if road_center is not None:
                car_commands.steer = self.compute_steering(road_center, FRAME_WIDTH)

            with self.lock:
                self.car_commands = car_commands
            # print("roadcenter", road_center)
            frame = self.visualize_curves(frame, left_lines, right_lines, FRAME_HEIGHT)
            frame = self.visualize_center(frame, road_center, FRAME_HEIGHT)

            img_canny_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            blended_frame = cv2.addWeighted(img_canny_color, 1, frame, 1, 0)
            self._notify_observers(blended_frame,timestamp = time.time())

    def read_inputs(self):
        with self.lock:
            return self.car_commands.copy()

    def denoise_frame(self, frame):
        denoised = cv2.GaussianBlur(frame, (3, 3), 1)
        return denoised

    def threshold_frame(self, frame):
        # _,img = cv2.threshold(frame,187,255,cv2.THRESH_BINARY)
        frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 11, 20)
        return frame

    def detect_edges(self, gray):
        return cv2.Canny(gray, 200, 150)

    def mask_region(self, image, height, width): #TODO height and width are wrong!?
        # Coordinates for the polygon covering the bottom half of the image
        #height, width = image.shape
        polygon = np.array([
            [(0, int(height*0.7)), (width, int(height*0.7)), (width, height), (0, height)]
        ])
        # Create a mask the same size as the image, initially black
        mask = np.zeros_like(image)
        # Fill the polygon area with white
        cv2.fillPoly(mask, [polygon], 255)

        # Apply the mask using bitwise AND to retain the bottom half
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image


    def detect_curves(self, img, image_height, image_width):
        # Convert image to binary if not already
        #_, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # Find all non-zero points in the image
        points = cv2.findNonZero(img)
        if points is not None:
            # Separate points into left and right based on x-coordinate
            mid_x = image_width // 2
            left_points = points[points[:, 0, 0] < mid_x]
            right_points = points[points[:, 0, 0] >= mid_x]

            # Fit polynomials if enough points are detected
            if left_points.shape[0] > 20:
                left_curve = np.polyfit(left_points[:, 0, 1], left_points[:, 0, 0], 2)
            else:
                left_curve = None

            if right_points.shape[0] > 20:
                right_curve = np.polyfit(right_points[:, 0, 1], right_points[:, 0, 0], 2)
            else:
                right_curve = None

            # Calculate road center at the bottom of the image using the average of the x-coordinates of the polynomial fits
            y_eval = image_height
            if left_curve is not None and right_curve is not None:
                left_x_bottom = np.polyval(left_curve, y_eval)
                right_x_bottom = np.polyval(right_curve, y_eval)
                road_center = (left_x_bottom + right_x_bottom) / 2
            else:
                road_center = None

            return left_curve, right_curve, road_center

        return None, None, None

    def visualize_curves(self, frame, left_curve, right_curve, image_height):
        plot_y = np.linspace(0, image_height - 1, image_height)
        colors = [(255, 0, 0), (0, 255, 0)]  # Red for left, Green for right

        curves = [left_curve, right_curve]
        for idx, curve in enumerate(curves):
            if curve is not None:
                plot_x = np.polyval(curve, plot_y)
                points = np.int32(np.vstack([plot_x, plot_y]).T)
                cv2.polylines(frame, [points], isClosed=False, color=colors[idx], thickness=5)

        return frame

    def compute_steering(self, road_center, frame_width):
        if road_center is None:
            return 0

        k_p = 0.9
        set_point = frame_width / 2  # desired setpoint is center of frame
        set_point_normalized = set_point / frame_width
        center_normalized = road_center / frame_width
        error = center_normalized - set_point_normalized

        return -error * k_p

    def visualize_center(self, frame, road_center, height):
        if road_center is not None:
            road_center = int(road_center)
            return cv2.circle(frame, (road_center, height), radius=10, color=(255, 0, 0), thickness=-1)
        else:
            return frame
