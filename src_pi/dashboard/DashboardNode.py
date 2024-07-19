import cv2
import zmq
import json
import numpy as np
from threading import Thread
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class SubscriberNode:
    def __init__(self, zmq_image_url="tcp://localhost:5555", zmq_gamepad_url="tcp://localhost:5556"):
        self.zmq_context = zmq.Context()

        # Initialize image subscriber
        self.zmq_image_subscriber = self.zmq_context.socket(zmq.SUB)
        self.zmq_image_subscriber.connect(zmq_image_url)
        self.zmq_image_subscriber.setsockopt_string(zmq.SUBSCRIBE, 'frame')

        # Initialize gamepad subscriber
        self.zmq_gamepad_subscriber = self.zmq_context.socket(zmq.SUB)
        self.zmq_gamepad_subscriber.connect(zmq_gamepad_url)
        self.zmq_gamepad_subscriber.setsockopt_string(zmq.SUBSCRIBE, 'gamepad')

        self.poller = zmq.Poller()
        self.poller.register(self.zmq_image_subscriber, zmq.POLLIN)
        self.poller.register(self.zmq_gamepad_subscriber, zmq.POLLIN)

        self.image = None
        self.gamepad_data = None

    def start(self):
        Thread(target=self.receive_data, daemon=True).start()
        self.visualize()

    def receive_data(self):
        while True:
            socks = dict(self.poller.poll())
            if self.zmq_image_subscriber in socks:
                topic, *msg_parts = self.zmq_image_subscriber.recv_multipart()
                if topic == b'frame':
                    self.image = cv2.imdecode(np.frombuffer(msg_parts[0], np.uint8), cv2.IMREAD_COLOR)
            if self.zmq_gamepad_subscriber in socks:
                message = self.zmq_gamepad_subscriber.recv_string()
                topic, payload = message.split(' ', 1)  # Split the topic and the JSON payload
                controller_data = json.loads(payload)
                timestamp = controller_data["timestamp"]
                self.gamepad_data = controller_data["data"]

    def draw_chart(self):
        fig, ax = plt.subplots(figsize=(4, 3))
        canvas = FigureCanvas(fig)
        ax.bar(['Throttle', 'Steering'],
               [self.gamepad_data['right_trigger'], self.gamepad_data['left_stick_y']],
               color=['blue', 'red'])
        ax.set_ylim([-1, 1])
        canvas.draw()

        buf = canvas.buffer_rgba()
        chart_img = np.asarray(buf, dtype=np.uint8)
        chart_img = cv2.cvtColor(chart_img, cv2.COLOR_RGBA2BGR)
        return chart_img

    def visualize(self):
        while True:
            if self.image is not None:
                # Resize image and chart to fit in the same window
                img_resized = cv2.resize(self.image, (640, 480))
                if self.gamepad_data:
                    chart_img = self.draw_chart()
                    chart_img_resized = cv2.resize(chart_img, (640, 480))
                    combined_img = np.hstack((img_resized, chart_img_resized))
                else:
                    combined_img = img_resized

                cv2.imshow('Camera Feed and Gamepad Data', combined_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()

    def stop(self):
        self.zmq_image_subscriber.close()
        self.zmq_gamepad_subscriber.close()
        self.zmq_context.term()
        print("Subscriber Node stopped and ZeroMQ subscribers closed.")


if __name__ == "__main__":
    subscriber_node = SubscriberNode(zmq_image_url="tcp://localhost:5555", zmq_gamepad_url="tcp://localhost:5556")
    subscriber_node.start()
