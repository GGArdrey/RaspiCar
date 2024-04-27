import time

import cv2
import numpy as np
from CarCommands import CarCommands
from IControlAlgorithm import IControlAlgorithm
import threading
from util import timer



class LaneDetectionHough(IControlAlgorithm):

    def __init__(self):
        super().__init__()
        super().__init__()
        self.lock = threading.Lock()
        self.new_frame_condition = threading.Condition(self.lock)
        self.latest_frame = None
        self.car_commands = CarCommands()
        self.processing_thread = threading.Thread(target=self.wait_and_process_frames)
        self.processing_thread.start()

    def update(self, frame):
        with self.new_frame_condition:
            self.latest_frame = frame
            self.new_frame_condition.notify()  # Signal that a new frame is available


    def read_inputs(self):
        with self.lock:
            return self.car_commands.copy()

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
        with timer("Process Frame Execution"):
            FRAME_WIDTH, FRAME_HEIGHT, _ = frame.shape
            car_commands = CarCommands()

            img = self.denoise_frame(frame)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = self.detect_edges(img)
            img = self.mask_region(img,FRAME_HEIGHT, FRAME_WIDTH)

            left_lines, right_lines, road_center = self.detect_lines(img, FRAME_HEIGHT, FRAME_WIDTH)
            car_commands.steer = self.compute_steering(road_center, FRAME_WIDTH)

            with self.lock:
                self.car_commands = car_commands
            # print("roadcenter", road_center)
            # frame = self.visualize_lines(frame, left_lines, right_lines) #TODO enable
            # frame = self.visualize_center(frame, road_center)
            # car_commands.steer = self.compute_steering(road_center)

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

    # def warp_perspective(self, img):
    #     height, width = img.shape
    #     src_points = np.float32([
    #         [int(width * 0.33), int(height * 0.5)],
    #         [int(width * 0.66), int(height * 0.5)],
    #         [int(width * 0.25), height],
    #         [int(width * 0.75), height]
    #     ])
    #
    #     ratio_offset = 100
    #     dst_points = np.float32([
    #         [ratio_offset, 0],
    #         [width - ratio_offset * 2, 0],
    #         [ratio_offset, height],
    #         [width - ratio_offset * 2, height]
    #     ])
    #
    #     warp_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    #     return cv2.warpPerspective(img, warp_matrix, (width, height))

    def mask_region(self, image, height, width):
        polygon = np.array([
            [(0, height), (0, height // 3), (width, height // 3), (width, height)]
        ])
        mask = np.zeros_like(image)
        mask = cv2.fillPoly(mask, polygon, 255)
        mask = cv2.bitwise_and(image, mask)
        return mask

    def detect_lines(self, img, image_height, image_width):
        left_lines = []
        right_lines = []
        left_x_bottom = []  # storing the intersect with the bottom of the image for all left lines
        right_x_bottom = []  # storing the intersect with the bottom of the image for all right lines
        road_center = None
        lines = cv2.HoughLinesP(img, rho=3, theta=np.pi / 180, threshold=50, minLineLength=30, maxLineGap=5)
        if lines is None:
            return [], [], road_center

        for line in lines:
            x1, y1, x2, y2 = line[0]
            theta = np.pi / 2 - np.arctan2(y2 - y1, x2 - x1)

            # Calculate the x-coordinate of the intersection point with the bottom of the image
            if np.cos(theta) > 0:
                right_lines.append([x1, y1, x2, y2])
                right_x_bottom.append(int(x1 + (image_height - y1) / np.tan(theta)))
            else:
                left_lines.append([x1, y1, x2, y2])
                left_x_bottom.append(int(x1 + (image_height - y1) / np.tan(theta)))

            # Calculate the road center
        if left_x_bottom and right_x_bottom:
            road_center = round((np.average(right_x_bottom) + np.average(left_x_bottom)) / 2)
        else:
            road_center = None

        return left_lines, right_lines, road_center

    def compute_steering(self, road_center, frame_width):
        if road_center is None:
            return 0

        k_p = 0.9
        set_point = frame_width / 2  # desired setpoint is center of frame
        set_point_normalized = set_point / frame_width
        center_normalized = road_center / frame_width
        error = center_normalized - set_point_normalized

        return error * k_p

    def visualize_lines(self, frame, left_lines, right_lines):
        # lines_visualize = np.zeros_like(frame)
        if left_lines:
            for x1, y1, x2, y2 in left_lines:
                frame = cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            left_average = np.average(left_lines, axis=0).astype(int)
            x1, y1, x2, y2 = left_average
            frame = cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

        if right_lines:
            for x1, y1, x2, y2 in right_lines:
                frame = cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            right_average = np.average(right_lines, axis=0).astype(int)
            x1, y1, x2, y2 = right_average
            frame = cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)

        # frame = cv2.addWeighted(frame, 1, lines_visualize, 1, 1)

        return frame

    def visualize_center(self, frame, road_center):
        if road_center is not None:
            return cv2.circle(frame, (road_center, self.FRAME_HEIGHT), radius=10, color=(255, 0, 0), thickness=-1)
        else:
            return frame
