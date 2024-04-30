import socket
import struct

import cv2
import threading
import numpy as np

class Client:
    def __init__(self, server_host='raspberrypi.local', server_port=8000):
        self.server_host = server_host
        self.server_port = server_port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def start(self):
        try:
            self.client_socket.connect((self.server_host, self.server_port))
            print("Connected to server.")
        except Exception as e:
            print(f"Connection failed: {e}")
            return

        # Start separate threads for keyboard input and video stream
        #keyboard_thread = threading.Thread(target=self.send_keyboard_input)
        video_thread = threading.Thread(target=self.receive_video_stream)

        #keyboard_thread.start()
        video_thread.start()

        #keyboard_thread.join()
        video_thread.join()

        self.client_socket.close()
        print("Client shut down.")

    # def send_keyboard_input(self):
    #     def on_press(key):
    #         try:
    #             key_data = str(key).encode()
    #             self.client_socket.sendall(key_data)
    #         except Exception as e:
    #             print(f"Error sending keyboard input: {e}")
    #
    #     listener = keyboard.Listener(on_press=on_press)
    #     listener.start()
    #     listener.join()

    def receive_video_stream(self):
        data = b""
        payload_size = struct.calcsize(">L")

        while True:
            while len(data) < payload_size:
                data += self.client_socket.recv(4096)

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]

            while len(data) < msg_size:
                data += self.client_socket.recv(4096)

            frame_data = data[:msg_size]
            data = data[msg_size:]

            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            cv2.imshow('Video Stream', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == '__main__':
    client = Client()
    client.start()