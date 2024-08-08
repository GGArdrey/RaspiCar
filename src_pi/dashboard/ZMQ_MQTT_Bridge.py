import zmq
import paho.mqtt.client as mqtt
import json

# Setup ZeroMQ subscriber
context = zmq.Context()
image_socket = context.socket(zmq.SUB)
image_socket.connect("tcp://localhost:5555")
image_socket.setsockopt_string(zmq.SUBSCRIBE, 'frame')

gamepad_socket = context.socket(zmq.SUB)
gamepad_socket.connect("tcp://localhost:5556")
gamepad_socket.setsockopt_string(zmq.SUBSCRIBE, 'gamepad')

# Setup MQTT client
mqtt_client = mqtt.Client()
mqtt_client.connect("localhost", 1883, 60)
mqtt_client.loop_start()

while True:
    socks = zmq.select([image_socket, gamepad_socket], [], [], 1.0)[0]
    for sock in socks:
        if sock == image_socket:
            topic, *msg_parts = image_socket.recv_multipart()
            if topic == b'frame':
                # Handle frame data
                mqtt_client.publish("dashboard/frame", msg_parts[0])
        elif sock == gamepad_socket:
            message = gamepad_socket.recv_string()
            topic, payload = message.split(' ', 1)
            controller_data = json.loads(payload)["data"]
            mqtt_client.publish("dashboard/gamepad", json.dumps(controller_data))
