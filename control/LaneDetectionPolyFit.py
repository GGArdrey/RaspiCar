import time

import cv2
import numpy as np
from CarCommands import CarCommands
from IControlAlgorithm import IControlAlgorithm
import threading
from util import timer
from IObservable import IObservable
from sklearn.cluster import KMeans, DBSCAN
np.random.RandomState(seed=123)
from filterpy.kalman import KalmanFilter
from sympy import symbols, diff, atan, N

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
        self.pid_controller = PIDController(0.02,0.0,0.05)
        # Initialize two Kalman Filters
        self.kf_left_lane = self.initialize_kalman()
        self.kf_right_lane = self.initialize_kalman()
        self.right_lane_prediction_counter = 0 #counts how often kalman was used back-to-back to predict right lane
        self.left_lane_prediction_counter = 0

    def start(self):
        if not self.running:
            print("Starting Lane Detection PolyFit Thread...")
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
        with timer("LaneDetectionPolyFit.process_frame Execution"):
            car_commands = CarCommands()

            img_color = self.resize_and_crop_image(frame)
            img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            img = self.denoise_frame(img)
            img = self.detect_edges(img)

            left_lines, right_lines, road_center, info = self.detect_curves_dbscan(img, 66, 200)

            # steering_deg = None
            # if left_lines is not None and right_lines is not None:
            #     steering_deg = self.compute_steering_curviture(left_lines,right_lines)


            if road_center is not None and info is not None:
                car_commands.steer = self.pid_controller.compute_steering(road_center, 200)
                # car_commands.steer = steering_deg/40
                car_commands.additional_info = info
                with self.lock:
                    self.car_commands = car_commands


            img_color = self.visualize_curves(img_color, left_lines, right_lines, 66)
            img_color = self.visualize_center(img_color, road_center, 66)
            img_canny_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img_color = cv2.addWeighted(img_canny_color, 1, img_color, 1, 0)
            self._notify_observers(img_color, timestamp=time.time())


    def compute_steering_curviture(self, left_coeffs, right_coeffs):
        # Example coefficients (replace with your actual values)
        # Assume a fixed look-ahead distance (e.g., 10 meters)
        look_ahead_distance = 55
        # Calculate the curvature for each lane
        left_curvature = left_coeffs[0] / (
                1 + (2 * left_coeffs[0] * look_ahead_distance + left_coeffs[1]) ** 2) ** (3 / 2)
        right_curvature = right_coeffs[0] / (
                1 + (2 * right_coeffs[0] * look_ahead_distance + right_coeffs[1]) ** 2) ** (3 / 2)
        # Compute the average radius of curvature
        radius = 1 / ((left_curvature + right_curvature) / 2)
        # Calculate the desired steering angle
        steering_angle = np.arctan((2 * look_ahead_distance) / (2 * radius))
        # Convert radians to degrees
        steering_angle_degrees = steering_angle * (180 / np.pi)

        #print(f"Steering angle in radians: {steering_angle:.4f}")
        print(f"Steering angle in degrees: {steering_angle_degrees:.4f}")
        return steering_angle_degrees

    def resize_and_crop_image(self, image, target_width=200, target_height=66):
        # Check input image dimensions
        if len(image.shape) < 2:
            raise ValueError("Invalid image data!")

        height, width = image.shape[:2]

        # Calculate scaling factor to maintain aspect ratio based on width
        scaling_factor = target_width / width
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        resized_image = cv2.resize(image, (new_width, new_height))

        # Check if the new height is greater than or equal to the target height before cropping
        if new_height < target_height:
            raise ValueError("Resized image height is less than the target crop height.")

        # Calculate start y-coordinate for cropping to center the crop area
        y_start = new_height - target_height

        cropped_image = resized_image[y_start:y_start + target_height, 0:target_width]
        #cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
        return cropped_image

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

    def cluster_points(self, img):
        points = cv2.findNonZero(img)
        if points is None:
            return None, None
        dbscan = DBSCAN(eps=15, min_samples=10).fit(points[:, 0, :])
        labels = dbscan.labels_
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        clusters = [points[labels == label] for label in unique_labels]
        return clusters, labels

    def fit_polynomials(self, clusters):
        if clusters is None:
            return None,None,None

        curve_mse_pairs = []

        for cluster in clusters:
            if len(cluster) > 10:
                curve = np.polyfit(cluster[:, 0, 1], cluster[:, 0, 0], 2)  # Fit a 2nd degree polynomial
                fit_x = np.polyval(curve, cluster[:, 0, 1])  # Evaluate polynomial at x positions of cluster
                mse = np.mean((cluster[:, 0, 0] - fit_x) ** 2)  # Calculate MSE
                curve_mse_pairs.append((curve, mse, cluster))
            else:
                curve_mse_pairs.append(
                    (None, float('inf'), cluster))  # Append None for curve and inf for MSE if not enough points

        # Sort curve-MSE pairs by MSE (the second item in each tuple)
        sorted_curve_mse_pairs = sorted(curve_mse_pairs, key=lambda x: x[1])
        if not sorted_curve_mse_pairs:  # Check if the list is empty
            return None, None, None  # Or any other appropriate default values

        # Unpack curves and MSEs from sorted pairs
        curves, mses, clusters = zip(*sorted_curve_mse_pairs)  # This will unzip the pairs into two tuples

        return curves, mses, clusters

    def select_lane_candidates(self, curves, mses, clusters, sizes):
        if curves is None or len(curves) == 0:
            return []

        # Define thresholds and epsilon to prevent division by zero
        epsilon = 1e-6
        alpha = 0.1  # Reduced impact of MSE in the score
        beta = 2  # Increased impact of size in the score
        size_threshold = 150  # Threshold for size scoring
        gamma = 2  # Exponential factor for size scoring
        score_thresh = 1

        # Compute scores for each candidate
        scored_candidates = [
            (curve, mse, cluster, size, alpha*-mse + beta * (
                (size / size_threshold) ** gamma if size > size_threshold else (size / size_threshold)))
            for curve, mse, cluster, size in zip(curves, mses, clusters, sizes)
        ]

        # Sort candidates by computed score (higher is better)
        scored_candidates.sort(key=lambda x: x[4], reverse=True)

        # Select up to two best candidates based on score
        # Ensuring that the best two scores are selected and that there are at least two good candidates
        self.print_results(scored_candidates)
        # if len(scored_candidates) >= 2 and scored_candidates[1][4] > score_thresh:
        #     best_candidates = scored_candidates[:2]
        # else:
        #     best_candidates = []

        return [[curve, mse, cluster, size, score] for curve, mse, cluster, size, score in scored_candidates]

    def calculate_road_center(self, curves, image_height):
        if curves[0] is not None and curves[1] is not None:
            y_eval = image_height - 1
            left_x_bottom = np.polyval(curves[0], y_eval)
            right_x_bottom = np.polyval(curves[1], y_eval)
            road_center = (left_x_bottom + right_x_bottom) / 2
            #print("road_center: ", road_center)
            return road_center
        return None



    def detect_curves_dbscan(self, img, image_height, image_width):
        clusters, labels = self.cluster_points(img)
        # if clusters is None or len(clusters) <= 0:
        #     return None, None, None, None


        if clusters is not None:
            clusters.sort(key=len, reverse=True)

        cluster_sizes = None
        curves, mses, clusters = self.fit_polynomials(clusters)
        if clusters is not None:
            cluster_sizes = [len(x) for x in clusters if x is not None]

        selected_lane_candidates = self.select_lane_candidates(curves, mses, clusters, cluster_sizes)

        lane1 = None
        lane2 = None
        info = None
        current_time = round(time.time() * 1000)
        if len(selected_lane_candidates) == 2:
            [[lane1, mse1, cluster1, size1, score1 ], [lane2, mse2, cluster2, size2, score2]] = selected_lane_candidates
            info = [current_time, mse1, mse2, size1, size2, score1, score2]
        elif len(selected_lane_candidates) > 2:
            first_two_lists = selected_lane_candidates[:2]
            [[lane1, mse1, cluster1, size1, score1], [lane2, mse2, cluster2, size2, score2]] = first_two_lists
            info = [current_time, mse1, mse2, size1, size2, score1, score2]
        else:
            info = [current_time, 0, 0, 0, 0, 0, 0]

        road_center = self.calculate_road_center([lane1, lane2], image_height)

        return (lane1, lane2, road_center, info)

    def print_results(self, selected_lane_candidates):
        # Header for the output
        print("{:<10} {:<10} {:<10}".format("MSE", "Size", "Score"))
        print("-" * 30)  # Print a simple line of dashes for separation

        # Loop through each candidate and print relevant details
        for candidate in selected_lane_candidates:
            mse = candidate[1]  # Assuming MSE is the second element in each sublist
            size = candidate[3]  # Assuming Size is the fourth element in each sublist
            score = candidate[4]  # Assuming Score is the fifth element in each sublist
            print(f"{mse:.3f}      {size:.0f}      {score:.3f}")
        print("-" * 30)  # Print a simple line of dashes for separation

    def visualize_curves(self, frame, left_curve, right_curve, image_height):
        average_curve = None
        if left_curve is not None and right_curve is not None:
            average_curve = (left_curve+right_curve)/2
        plot_y = np.linspace(0, image_height - 1, image_height)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red for left, Green for right

        curves = [left_curve, right_curve] #average_curve
        for idx, curve in enumerate(curves):
            if curve is not None:
                plot_x = np.polyval(curve, plot_y)
                points = np.int32(np.vstack([plot_x, plot_y]).T)
                cv2.polylines(frame, [points], isClosed=False, color=colors[idx], thickness=5)


        return frame


    def visualize_center(self, frame, road_center, height):
        if road_center is not None:
            road_center = int(road_center)
            return cv2.circle(frame, (road_center, 66), radius=5, color=(0, 0, 255), thickness=-1) #TODO 200 magic number
        else:
            return frame


class PIDController:
    def __init__(self, kp, ki, kd, max_error=100):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_error = max_error
        self.integral = 0
        self.prev_error = 0

    def compute_steering(self, lane_center, frame_width):
        if lane_center is None:
            return 0

        set_point = frame_width / 2  # desired setpoint is center of frame
        error = lane_center - set_point
        error = max(min(error, self.max_error), -self.max_error)  # Clamp error to max_error

        self.integral += error
        derivative = error - self.prev_error

        steering = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        if steering > 1:
            steering = 1

        if steering < -1:
            steering = -1


        self.prev_error = error

        return steering