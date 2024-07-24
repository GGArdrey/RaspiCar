import cv2
import numpy as np
import zmq
import logging
from utils.timer_utils import timer
from utils.message_utils import (
    create_json_message,
    parse_image_message,
    create_compressed_image_message
)
from Node import Node
from filterpy.kalman import KalmanFilter
from sklearn.cluster import DBSCAN


class LaneDetectionNode(Node):
    def __init__(self, zmq_pub_url="tcp://*:5561",
                 zmq_pub_topic="lane_detection_steering_commands",
                 zmq_camera_pub_url="tcp://*:5550",
                 zmq_camera_pub_topic="camera_lane_detection",
                 camera_sub_url="tcp://*:5555",
                 camera_sub_topic="camera",
                 log_level=logging.INFO):
        super().__init__(log_level=log_level)
        self.latest_frame = None
        self.pid_controller = PIDController(0.02, 0.0, 0.05)

        # ZeroMQ setup
        self.zmq_pub_url = zmq_pub_url
        self.zmq_pub_topic = zmq_pub_topic
        self.camera_sub_url = camera_sub_url
        self.camera_sub_topic = camera_sub_topic

        self.zmq_context = zmq.Context()
        self.zmq_publisher = self.zmq_context.socket(zmq.PUB)
        self.zmq_publisher.bind(self.zmq_pub_url)

        self.zmq_camera_pub_url = zmq_camera_pub_url
        self.pub_camera_topic = zmq_camera_pub_topic
        self.zmq_camera_publisher = self.zmq_context.socket(zmq.PUB)
        self.zmq_camera_publisher.setsockopt(zmq.SNDHWM, 1)  # Set high water mark to 1 to drop old frames
        self.zmq_camera_publisher.bind(self.zmq_camera_pub_url)

        self.zmq_subscriber = self.zmq_context.socket(zmq.SUB)
        self.zmq_subscriber.connect(self.camera_sub_url)
        self.zmq_subscriber.setsockopt(zmq.RCVHWM, 1)
        self.zmq_subscriber.setsockopt_string(zmq.SUBSCRIBE, self.camera_sub_topic)

    def start(self):
        while True:
            try:
                message = self.zmq_subscriber.recv_multipart()
                topic, image, timestamp = parse_image_message(message)
                self.process_frame(image, timestamp)
            except zmq.ZMQError as e:
                self.log(f"ZMQ error: {e}", logging.ERROR)
            except Exception as e:
                self.log(f"Error Processing Frame: {e}", logging.ERROR)

    def release(self):
        self.zmq_publisher.close()
        self.zmq_subscriber.close()
        self.zmq_context.term()

    def process_frame(self, frame, timestamp):
        with timer("LaneDetectionNode.process_frame Execution"):
            img_color = self.resize_and_crop_image(frame)
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            img_denoised = self.denoise_frame(img_gray)
            img_edges = self.detect_edges(img_denoised)

            left_lines, right_lines, road_center, info = self.detect_curves_dbscan(img_edges, 66, 200)

            steering_commands = self.generate_steering_commands(road_center)

            self.log(f"Steering Prediction: {steering_commands['steer']}", logging.DEBUG)
            self.publish_steering_commands(steering_commands, timestamp)

            img_color = self.overlay_visuals(img_color, img_edges, left_lines, right_lines, road_center)

            self.publish_camera_frame(img_color, timestamp)

    def generate_steering_commands(self, road_center):
        commands = {
            "steer": 0,
            "throttle": 0,
            "emergency_stop": 0,
            "reset_emergency_stop": 0,
            "sensors_enable": 0,
            "sensors_disable": 0
        }

        if road_center is not None:
            commands["steer"] = self.pid_controller.compute_steering(road_center, 200)

        return commands

    def publish_steering_commands(self, commands, timestamp):
        message = create_json_message(commands, self.zmq_pub_topic, timestamp=timestamp)
        self.zmq_publisher.send(message)

    def publish_camera_frame(self, frame, timestamp):
        message = create_compressed_image_message(frame, self.pub_camera_topic, timestamp)
        self.zmq_camera_publisher.send_multipart(message, flags=zmq.NOBLOCK)

    def resize_and_crop_image(self, image, target_width=200, target_height=66):
        height, width = image.shape[:2]
        scaling_factor = target_width / width
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        resized_image = cv2.resize(image, (new_width, new_height))
        y_start = new_height - target_height
        return resized_image[y_start:y_start + target_height, 0:target_width]

    def denoise_frame(self, frame):
        return cv2.GaussianBlur(frame, (3, 3), 1)

    def detect_edges(self, gray):
        return cv2.Canny(gray, 200, 150)

    def cluster_points(self, img):
        points = cv2.findNonZero(img)
        if points is None:
            return None, None
        dbscan = DBSCAN(eps=5, min_samples=10).fit(points[:, 0, :])
        labels = dbscan.labels_
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        clusters = [points[labels == label] for label in unique_labels]
        return clusters, labels

    def fit_polynomials(self, clusters):
        if clusters is None:
            return None, None, None
        curve_mse_pairs = []
        for cluster in clusters:
            if len(cluster) > 10:
                curve = np.polyfit(cluster[:, 0, 1], cluster[:, 0, 0], 2)
                fit_x = np.polyval(curve, cluster[:, 0, 1])
                mse = np.mean((cluster[:, 0, 0] - fit_x) ** 2)
                curve_mse_pairs.append((curve, mse, cluster))
            else:
                curve_mse_pairs.append((None, float('inf'), cluster))
        sorted_curve_mse_pairs = sorted(curve_mse_pairs, key=lambda x: x[1])
        if not sorted_curve_mse_pairs:
            return None, None, None
        curves, mses, clusters = zip(*sorted_curve_mse_pairs)
        return curves, mses, clusters

    def select_lane_candidates(self, curves, mses, clusters, sizes):
        if curves is None or len(curves) == 0:
            return []
        epsilon = 1e-6
        alpha = 0.1
        beta = 2
        size_threshold = 150
        gamma = 2
        scored_candidates = [
            (curve, mse, cluster, size, alpha * -mse + beta * (
                (size / size_threshold) ** gamma if size > size_threshold else (size / size_threshold)))
            for curve, mse, cluster, size in zip(curves, mses, clusters, sizes)
        ]
        scored_candidates.sort(key=lambda x: x[4], reverse=True)
        return [[curve, mse, cluster, size, score] for curve, mse, cluster, size, score in scored_candidates]

    def calculate_road_center(self, curves, image_height):
        if curves[0] is not None and curves[1] is not None:
            y_eval = image_height - 1
            left_x_bottom = np.polyval(curves[0], y_eval)
            right_x_bottom = np.polyval(curves[1], y_eval)
            return (left_x_bottom + right_x_bottom) / 2
        return None

    def detect_curves_dbscan(self, img, image_height, image_width):
        clusters, labels = self.cluster_points(img)
        if clusters is not None:
            clusters.sort(key=len, reverse=True)
        curves, mses, clusters = self.fit_polynomials(clusters)
        cluster_sizes = [len(x) for x in clusters if x is not None]
        selected_lane_candidates = self.select_lane_candidates(curves, mses, clusters, cluster_sizes)

        self.print_results(selected_lane_candidates)

        if len(selected_lane_candidates) < 2:
            return None, None, None, None

        [[lane1, mse1, cluster1, size1, score1], [lane2, mse2, cluster2, size2, score2]] = selected_lane_candidates[:2]

        road_center = self.calculate_road_center([lane1, lane2], image_height)
        return lane1, lane2, road_center, [mse1, mse2, size1, size2, score1, score2]

    def overlay_visuals(self, img_color, img_edges, left_lines, right_lines, road_center):
        img_color = self.visualize_curves(img_color, left_lines, right_lines, 66)
        img_color = self.visualize_center(img_color, road_center, 66)
        img_canny_color = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(img_canny_color, 1, img_color, 1, 0)

    def visualize_curves(self, frame, left_curve, right_curve, image_height):
        average_curve = None
        if left_curve is not None and right_curve is not None:
            average_curve = (left_curve + right_curve) / 2
        plot_y = np.linspace(0, image_height - 1, image_height)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red for left, Green for right

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
            return cv2.circle(frame, (road_center, 66), radius=5, color=(0, 0, 255), thickness=-1)
        else:
            return frame

    def print_results(self, selected_lane_candidates):
        # Header for the output
        self.log("{:<10} {:<10} {:<10}".format("MSE", "Size", "Score"), logging.DEBUG)
        self.log("-" * 30, logging.DEBUG)  # Print a simple line of dashes for separation

        # Loop through each candidate and print relevant details
        for candidate in selected_lane_candidates:
            mse = candidate[1]  # Assuming MSE is the second element in each sublist
            size = candidate[3]  # Assuming Size is the fourth element in each sublist
            score = candidate[4]  # Assuming Score is the fifth element in each sublist
            self.log(f"{mse:.3f}      {size:.0f}      {score:.3f}", logging.DEBUG)
        self.log("-" * 30, logging.DEBUG)  # Print a simple line of dashes for separation


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
        set_point = frame_width / 2
        error = lane_center - set_point
        error = max(min(error, self.max_error), -self.max_error)
        self.integral += error
        derivative = error - self.prev_error
        steering = (
                self.kp * error
                + self.ki * self.integral
                + self.kd * derivative
        )
        steering = max(min(steering, 1), -1)
        self.prev_error = error
        return steering


if __name__ == "__main__":
    lane_detection = LaneDetectionNode()
    lane_detection.start()
