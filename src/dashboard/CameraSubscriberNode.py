import cv2
import zmq
import numpy as np
import time

class CameraSubscriberNode:
    def __init__(self, zmq_sub_url="tcp://localhost:5555"):
        self.zmq_sub_url = zmq_sub_url
        self.zmq_context = zmq.Context()
        self.zmq_subscriber = self.zmq_context.socket(zmq.SUB)
        self.zmq_subscriber.connect(self.zmq_sub_url)
        self.zmq_subscriber.setsockopt_string(zmq.SUBSCRIBE, 'frame')

    def start(self):
        while True:
            try:
                topic, frame_data, timestamp = self.zmq_subscriber.recv_multipart()
                np_frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
                if frame is not None:
                    cv2.imshow('Received Camera Feed', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                        break
                else:
                    print("Camera Subscriber Node: Error decoding frame.")
            except Exception as e:
                print(f"Camera Subscriber Node: Error receiving frame: {e}")
                break

    def release(self):
        self.zmq_subscriber.close()
        self.zmq_context.term()
        cv2.destroyAllWindows()
        print("Camera Subscriber Node released and ZeroMQ subscriber closed.")

if __name__ == "__main__":
    subscriber_node = CameraSubscriberNode()
    try:
        subscriber_node.start()
    except KeyboardInterrupt:
        pass
    finally:
        subscriber_node.release()
