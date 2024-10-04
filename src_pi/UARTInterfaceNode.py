"""
RaspiCar
Copyright (c) 2024 Fynn Luca Maaß

Licensed under the Custom License. See the LICENSE file in the project root for license terms.
"""

import serial
import time
import threading
from threading import Lock
import zmq
import logging
from utils.message_utils import create_json_message, parse_json_message
from Node import Node


class UARTInterfaceNode(Node):
    def __init__(self, log_level=logging.INFO,
                 zmq_pub_url="tcp://localhost:5580",
                 pub_topic="pico_data",
                 control_sub_url="tcp://localhost:5570",
                 control_sub_topic="fused_steering_commands"):
        super().__init__(log_level=log_level)

        # UART setup
        port = "/dev/ttyS0"
        baudrate = 115200
        self.keep_running = True

        try:
            self.uart = serial.Serial(port, baudrate)
            self.lock = Lock()
        except serial.serialutil.SerialException as e:
            self.log(f"Could not open UART port: {e}", logging.ERROR)
            self.uart = None

        # ZeroMQ Publisher Setup
        self.zmq_context = zmq.Context()
        self.zmq_publisher = self.zmq_context.socket(zmq.PUB)
        self.zmq_publisher.bind(zmq_pub_url)
        self.zmq_pub_topic = pub_topic

        # ZeroMQ Subscriber for Control Commands
        self.control_subscriber = self.zmq_context.socket(zmq.SUB)
        self.control_subscriber.connect(control_sub_url)
        self.control_subscriber.setsockopt_string(zmq.SUBSCRIBE, control_sub_topic)

    def start(self):
        # Start threads
        self.keep_running = True
        threading.Thread(target=self.send_ping, daemon=True).start()
        threading.Thread(target=self.read_pico_data, daemon=True).start()
        threading.Thread(target=self.process_control_commands, daemon=True).start()

        # Keep the start method blocking until stop() is called
        try:
            while self.keep_running:
                time.sleep(0.1)  # Sleep to prevent excessive CPU usage
        except KeyboardInterrupt:
            self.log("Keyboard interrupt received. Shutting down.", logging.INFO)
            self.keep_running = False

    def stop(self):
        self.keep_running = False

    def release(self):
        self.stop()  # Signal threads to stop
        if self.uart:
            self.uart.close()
        self.zmq_publisher.close()
        self.control_subscriber.close()
        self.zmq_context.term()
        self.log("Resources released for UARTInterfaceNode.", logging.INFO)

    def send_ping(self):
        while True:
            try:
                if self.uart:
                    with self.lock:
                        self.uart.write(('ping,1\n').encode('utf8'))
                time.sleep(0.1)
            except Exception as e:
                self.log(f"Error in send_ping: {e}", logging.ERROR)


    def read_pico_data(self):
        while True:
            try:
                if self.uart:
                    with self.lock:
                        line = self.uart.readline().decode('utf-8').strip()
                        if line:
                            # Split by commas to get each key-value pair
                            pairs = line.split(',')
                            # Create a dictionary by splitting each pair at the colon
                            data_dict = {key: float(value) for key, value in (pair.split(':') for pair in pairs)}

                            self.publish_pico_data(data_dict)
                            self.log(f"Pico Data: {line}", logging.DEBUG)
                time.sleep(0.01)
            except Exception as e:
                self.log(f"Error in read_pico_data: {e}", logging.ERROR)


    def process_control_commands(self):
        while True:
            try:
                message = self.control_subscriber.recv_string()
                _, _, payload = parse_json_message(message)
                self.execute_uart_commands(payload)
            except Exception as e:
                self.log(f"Error in process_control_commands: {e}", logging.ERROR)


    def publish_pico_data(self, data):
        message = create_json_message(data, self.zmq_pub_topic)
        self.zmq_publisher.send(message)



    def execute_uart_commands(self, message):
        if "steer" in message:
            self.steer(message["steer"])
        if "throttle" in message:
            self.throttle(message["throttle"])
        if "sensors_enable" in message and message["sensors_enable"]:
            self.sensors_enable()
        if "sensors_disable" in message and message["sensors_disable"]:
            self.sensors_disable()
        if "emergency_stop" in message and message["emergency_stop"]:
            self.emergency_stop()
        if "reset_emergency_stop" in message and message["reset_emergency_stop"]:
            self.reset_emergency_stop()

    def sensors_enable(self):
        if self.uart:
            with self.lock:
                self.uart.write(('sensors_enable,1\n').encode('utf8'))

    def sensors_disable(self):
        if self.uart:
            with self.lock:
                self.uart.write(('sensors_disable,1\n').encode('utf8'))

    def emergency_stop(self):
        if self.uart:
            with self.lock:
                self.uart.write(('emergency_stop,1\n').encode('utf8'))

    def reset_emergency_stop(self):
        if self.uart:
            with self.lock:
                self.uart.write(('reset_emergency_stop,1\n').encode('utf8'))

    def throttle(self, value):
        value = value * 100
        if self.uart:
            with self.lock:
                self.uart.write(('throttle,' + str(int(value)) + '\n').encode('utf-8'))

    def steer(self, value):
        value = value * 100
        if self.uart:
            with self.lock:
                self.uart.write(('steer,' + str(int(value)) + '\n').encode('utf-8'))


# Example usage
if __name__ == "__main__":
    ci = UARTInterfaceNode()
