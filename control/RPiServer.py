import select
import socket
import struct
import threading

import cv2
from IObserver import Observer
#from control.Webcam import FrameCapture

class RPiServer(Observer):
    def __init__(self, keyboard_input=None, host='', port=8000):
        #self.keyboard_input = keyboard_input
        super().__init__()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((host, port))
        self.server_socket.listen(1)
        self.lock = threading.Lock()
        self.clients = []
        print(f'Server listening on {host}:{port}')

    def start_accepting_connections(self):
        '''
        Listens and accepts incoming connections. Those connections are stored inside this server class.
        This function is meant to be run in a thread, as it blocks while listening for incoming connections
        :return:
        '''
        while True:
            print('Waiting for a connection...')
            client_socket, addr = self.server_socket.accept()
            #readable, _, _ = select.select(inputs, outputs, inputs, 1)  # using select as non-blocking I/O
            print(f'Connected with {addr}')
            with self.lock:
                self.clients.append(client_socket)


    # def handle_keyboard_input(self, conn):
    #     while True:
    #         data = conn.recv(1024)
    #         if not data:
    #             break
    #         # Process the received keyboard input data
    #         keyboard_commands = data.decode()
    #         self.keyboard_input.process_keyboard_commands(keyboard_commands)

    def update(self, frame):
        '''
        Sends the received image to all clients
        :param frame: OpenCV image from the observable object
        :return:
        '''
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
        with self.lock:
            for client in self.clients:
                try:
                    frame_size = len(encoded_frame)
                    client.sendall(struct.pack('>L', frame_size) + encoded_frame.tobytes())
                except Exception as e:
                    print(f"Error sending video to client: {e}")
                    self.clients.remove(client)
                    client.close()
