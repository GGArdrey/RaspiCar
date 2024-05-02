
import socket
import struct
import threading

import cv2
from IObserver import IObserver


# from control.Webcam import FrameCapture

class RPiServer(IObserver):
    def __init__(self, keyboard_input=None, host='', port=8000):
        # self.keyboard_input = keyboard_input
        super().__init__()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((host, port))
        self.server_socket.listen(1)
        self.lock = threading.Lock()
        self.clients = []
        self.frame_condition = threading.Condition(self.lock)
        self.running = False
        self.latest_frame_data = None  # Storing latest camera image
        self.accept_thread = None
        self.send_thread = None
        print(f'Server listening on {host}:{port}')

    def start(self):
        if not self.running:
            print("Starting Server Accepting and Sending Threads")
            self.running = True
            self.server_socket.listen(1)
            self.accept_thread = threading.Thread(target=self.start_accepting_connections)
            self.send_thread = threading.Thread(target=self.send_frames_to_clients)
            self.accept_thread.start()
            self.send_thread.start()

    def stop(self):
        if self.running:
            self.running = False
            if self.accept_thread:
                self.accept_thread.join()
            if self.send_thread:
                self.send_thread.join()

            for client in self.clients:
                client.close()
            self.server_socket.close()
            print("Server stopped and all connections closed.")

    def start_accepting_connections(self):
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                print(f'Connected with {addr}')
                with self.lock:
                    self.clients.append(client_socket)
            except socket.error as e:
                if self.running:  # To distinguish expected stop errors from actual errors
                    print(f"Accept error: {e}")
                break

    def update(self, frame, timestamp):
        '''
        Takes the latest frame from the camera and updates latest_frame_data. Also notifies thread waiting on that data
        :param timestamp:
        :param frame:
        :return:
        '''
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
        with self.frame_condition:
            self.latest_frame_data = (encoded_frame.tobytes(), len(encoded_frame))
            self.frame_condition.notify_all()  # Notify all waiting threads that a new frame is available

    def send_frames_to_clients(self):
        while True:
            with self.frame_condition:
                self.frame_condition.wait()  # Wait until a frame is ready
                if self.latest_frame_data is None:
                    continue
                frame_bytes, frame_size = self.latest_frame_data

            # Copy the list of clients to avoid holding the lock while sending data
            with self.lock:
                clients = self.clients.copy()

            for client in clients:
                try:
                    client.sendall(struct.pack('>L', frame_size) + frame_bytes)
                except Exception as e:
                    print(f"Error sending video to client: {e}")
                    with self.lock:
                        self.clients.remove(client)
                        client.close()

    def stop_server(self):
        with self.frame_condition:
            self.latest_frame_data = None
            self.frame_condition.notify_all()  # Wake up the thread if it's waiting, to allow it to exit

        # Properly close all client sockets and the server socket
        with self.lock:
            for client in self.clients:
                client.close()
            self.server_socket.close()
