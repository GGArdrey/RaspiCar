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
        self.pid_controller = PIDController(0.02,0,0)
        # Initialize two Kalman Filters
        self.kf_left_lane = self.initialize_kalman()
        self.kf_right_lane = self.initialize_kalman()

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
        with timer("LaneDetectionPolyFit.process_frame Execution"):
            car_commands = CarCommands()

            img_color = self.resize_and_crop_image(frame)
            img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            img = self.denoise_frame(img)
            img = self.detect_edges(img)

            FRAME_WIDTH, FRAME_HEIGHT = img.shape
            left_lines, right_lines, road_center, colored_clusters_img = self.detect_curves_dbscan(img, 66, 200)


            # Compute steering based on road center
            FRAME_WIDTH, FRAME_HEIGHT = img.shape
            #print("center: ", road_center)
            if road_center is not None:
                car_commands.steer = self.pid_controller.compute_steering(road_center, 200)
                #print("Steer Lane Detection: ", car_commands.steer)

            with self.lock:
                self.car_commands = car_commands

            img_color = self.visualize_curves(img_color, left_lines, right_lines, 66)
            img_color = self.visualize_center(img_color, road_center, 66)


            #img_canny_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if colored_clusters_img is not None:
                img_color = cv2.addWeighted(colored_clusters_img, 1, img_color, 1, 0)
            self._notify_observers(img_color,timestamp = time.time())


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

        # Unpack curves and MSEs from sorted pairs
        curves, mses, clusters = zip(*sorted_curve_mse_pairs)  # This will unzip the pairs into two tuples

        return curves, mses, clusters

    def select_lane_candidates(self, curves, mses, clusters, sizes):
        # Define thresholds and epsilon to prevent division by zero
        score_thresh = -2.5
        mse_threshold = 50
        min_cluster_size = 50
        epsilon = 1e-6
        alpha = 1
        beta =1

        # Filter out based on MSE and size
        filtered_candidates = [
            (curve, mse, cluster, size) for curve, mse, cluster, size in zip(curves, mses, clusters, sizes)
            if mse < mse_threshold and size > min_cluster_size
        ]

        if not filtered_candidates:
            return []

        # Extract max size for normalization
        max_size = max(size for _, _, _, size in filtered_candidates)

        # Compute scores for each candidate (assumed alpha = 1, beta = 1 for simplicity)
        scored_candidates = [
            (curve, mse, cluster, size, alpha*np.log(1 / (mse + epsilon)) + beta*(size / max_size))
            for curve, mse, cluster, size in filtered_candidates
        ]

        # Filter out based on Score
        filtered_candidates = [
            (curve, mse, cluster, size, score) for curve, mse, cluster, size, score in scored_candidates
            if score > score_thresh
        ]

        # Sort candidates by computed score (higher is better)
        filtered_candidates.sort(key=lambda x: x[4], reverse=True)

        #print("MSE, Size, Score",[(mse, size, score) for curve, mse, cluster, size, score in filtered_candidates])

        # Select up to two best candidates based on score
        #selected_candidates = scored_candidates[:2]

        return [[cluster, curve, score] for curve, mse, cluster, size, score in filtered_candidates] #, score)

    def calculate_road_center(self, curves, image_height):
        if curves[0] is not None and curves[1] is not None:
            y_eval = image_height - 1
            left_x_bottom = np.polyval(curves[0], y_eval)
            right_x_bottom = np.polyval(curves[1], y_eval)
            road_center = (left_x_bottom + right_x_bottom) / 2
            #print("road_center: ", road_center)
            return road_center
        return None

    def color_clusters(self, img, labels):
        points = cv2.findNonZero(img)
        # Convert grayscale image to color for visualization
        colored_clusters = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Define and generate colors for each unique label
        unique_labels = set(labels)
        colors = {label: list(np.random.randint(0, 255, 3)) for label in unique_labels if label != -1}
        colors[-1] = [0, 0, 0]  # Black color for noise
        # Assign colors to each cluster point
        for point, label in zip(points[:, 0, :], labels):
            colored_clusters[point[1], point[0]] = colors[label]
        return colored_clusters  # Make sure to return the colored image

    def detect_curves_dbscan(self, img, image_height, image_width):
        clusters, labels = self.cluster_points(img)
        if clusters is None or len(clusters) < 0:
            return None, None, None, None
        clusters.sort(key=len, reverse=True)


        curves, mses, clusters = self.fit_polynomials(clusters)
        cluster_sizes = [len(x) for x in clusters if x is not None]
        selected_lane_candidates = self.select_lane_candidates(curves, mses, clusters, cluster_sizes)

        # Get predictions from the Kalman filters for both lanes
        predicted_left, predicted_right = self.predict_lanes()
        # print("Prediction L: ", predicted_left)
        # print("Prediction R: ", predicted_right)
        left_lane = None
        right_lane = None
        if not selected_lane_candidates:
            left_lane = predicted_left
            right_lane = predicted_right
            print("Subsitute both lanes by kalman prediction")
        elif len(selected_lane_candidates) == 2:
            # Case 1: Found two lanes and their score is very good - update Kalman Filters:
            [[cluster1, lane1, lane_score1], [cluster2, lane2, lane_score2]] = selected_lane_candidates
            x_bottom1 = np.polyval(lane1, image_height)
            x_bottom2 = np.polyval(lane2, image_height)
            # print("Lane1:", lane1)
            # print("Lane2:", lane2)
            if x_bottom1 < x_bottom2: #determine right and left lane
                left_lane = lane1
                right_lane = lane2
                self.update_kalman_filter(self.kf_left_lane, lane1)
                self.update_kalman_filter(self.kf_right_lane, lane2)
            else:
                left_lane = lane2
                right_lane = lane1
                self.update_kalman_filter(self.kf_left_lane, lane2)
                self.update_kalman_filter(self.kf_right_lane, lane1)
        else:
            # Case 2: Found more than two or one lane and their score is very good - use Kalman Filters to determine
            # most likely candidates:
            lanes = [item[1] for item in selected_lane_candidates] # unpack to get only the curves
            # Identify and select the most plausible lanes from the candidates
            left_lane, right_lane = self.select_lane_candidates_using_kalman(lanes, predicted_left, predicted_right)
            if left_lane is not None: # Update Kalman, since a left lane was found which aligns with the prediction
                self.update_kalman_filter(self.kf_left_lane, left_lane)
            else:
                left_lane = predicted_left
                print("Subsitute left lane by kalman prediction")

            if right_lane is not None: # Update Kalman, since a right lane was found which aligns with the prediction
                self.update_kalman_filter(self.kf_right_lane, right_lane)
            else:
                right_lane = predicted_right
                print("Subsitute right lane by kalman prediction")


        # Calculate the road center based on the selected or predicted lanes
        road_center = self.calculate_road_center([left_lane, right_lane], image_height)
        img_colored = self.color_clusters(img, labels)

        return (left_lane, right_lane, road_center, img_colored)


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


    def visualize_center(self, frame, road_center, height):
        if road_center is not None:
            road_center = int(road_center)
            return cv2.circle(frame, (road_center, 66), radius=5, color=(0, 0, 255), thickness=-1) #TODO 200 magic number
        else:
            return frame

    def select_lane_candidates_using_kalman(self, detected_curves, predicted_left, predicted_right):
        validated_lanes = []
        for curve in detected_curves:
            dist_to_left = np.linalg.norm(np.array(curve) - np.array(predicted_left))
            dist_to_right = np.linalg.norm(np.array(curve) - np.array(predicted_right))
            if dist_to_left < dist_to_right:
                validated_lanes.append(('left', curve, dist_to_left))
            else:
                validated_lanes.append(('right', curve, dist_to_right))

        left_candidates = [lane for lane in validated_lanes if lane[0] == 'left']
        right_candidates = [lane for lane in validated_lanes if lane[0] == 'right']

        left_lane = min(left_candidates, key=lambda x: x[2])[1] if left_candidates else None
        right_lane = min(right_candidates, key=lambda x: x[2])[1] if right_candidates else None

        return left_lane, right_lane

    def initialize_kalman(self):
        kf = KalmanFilter(dim_x=6, dim_z=3)  # 6 for [a, a_dot, b, b_dot, c, c_dot], 3 for [a, b, c]
        dt = 1  # time step between measurements
        kf.F = np.array([[1, dt, 0, 0, 0, 0],  # state transition model
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, dt, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, dt],
                         [0, 0, 0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0, 0, 0],  # measurement function
                         [0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0]])
        # Measurement uncertainty (how much do we trust the measurements?)
        kf.R = np.eye(3) * 0.5  # relatively high trust

        # Process uncertainty (how much do we trust our model prediction?)
        kf.Q = np.eye(6) * 0.1  # moderate trust in the constancy of velocity

        # Initial estimation error covariance
        kf.P *= 1000.  # start with a large uncertainty
        return kf

    def update_kalman_filter(self, kf, lane):
        if lane is not None and all(value is not None for value in lane):
            kf.update(lane)

    def predict_lanes(self):
        self.kf_left_lane.predict()
        self.kf_right_lane.predict()
        left_lane = self.kf_left_lane.x[[0, 2, 4]]# Extract coefficients [a, b, c]
        right_lane = self.kf_right_lane.x[[0, 2, 4]]
        return (left_lane.flatten(),right_lane.flatten())



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

        self.prev_error = error

        return steering