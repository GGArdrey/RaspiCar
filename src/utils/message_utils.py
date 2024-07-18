import json
import time
import cv2
import numpy as np

def create_json_message(payload, topic):
    message = {
        "timestamp": time.time(),
        "payload": payload
    }
    marshalled_message = f"{topic} {json.dumps(message)}".encode('utf-8')
    return marshalled_message

def parse_json_message(message):
    #decoded_message = message.decode('utf-8')
    topic, json_part = message.split(' ', 1)
    parsed_message = json.loads(json_part)
    timestamp = parsed_message["timestamp"]
    payload = parsed_message["payload"]
    return topic, timestamp, payload

def create_image_message(image, topic):
    _, buffer = cv2.imencode('.jpg', image) # Note: that we are encoding the image to JPEG format, maybe PNG?
    timestamp = str(time.time()).encode('utf-8')
    marshalled_message = [topic.encode('utf-8'), buffer.tobytes(), timestamp]
    return marshalled_message

def parse_image_message(message):
    topic = message[0].decode('utf-8')
    image_data = np.frombuffer(message[1], dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    timestamp = float(message[2].decode('utf-8'))
    return topic, image, timestamp
