import json
import time
import numpy as np
import cv2


def ensure_json_serializable(value):
    if isinstance(value, np.ndarray):
        return {'__ndarray__': value.tolist()}
    elif isinstance(value, (np.float32, np.float64)):
        return {'__float__': float(value)}
    elif isinstance(value, (np.int32, np.int64)):
        return {'__int__': int(value)}
    elif isinstance(value, (list, tuple)):
        return [ensure_json_serializable(item) for item in value]
    elif isinstance(value, dict):
        return {k: ensure_json_serializable(v) for k, v in value.items()}
    else:
        return value


def restore_original_type(value):
    if isinstance(value, dict):
        if '__ndarray__' in value:
            return np.array(value['__ndarray__'])
        elif '__float__' in value:
            return float(value['__float__'])
        elif '__int__' in value:
            return int(value['__int__'])
        else:
            return {k: restore_original_type(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [restore_original_type(item) for item in value]
    else:
        return value


def create_json_message(payload, topic, timestamp=None):
    if timestamp is None:
        timestamp = time.time()

    serializable_payload = ensure_json_serializable(payload)
    message = {
        "timestamp": timestamp,
        "payload": serializable_payload
    }
    marshalled_message = f"{topic} {json.dumps(message)}".encode('utf-8')
    return marshalled_message


def parse_json_message(message):
    topic, json_part = message.split(' ', 1)
    parsed_message = json.loads(json_part)
    timestamp = float(parsed_message["timestamp"])
    payload = restore_original_type(parsed_message["payload"])
    return topic, timestamp, payload


def create_image_message(image, topic, timestamp=None):
    if timestamp is None:
        timestamp = time.time()
    timestamp = str(timestamp).encode('utf-8')
    marshalled_message = [topic.encode('utf-8'), image.tobytes(), timestamp]
    return marshalled_message

def parse_image_message(message, frame_height=640, frame_width=360):
    topic, image_data, timestamp = message
    image = np.frombuffer(image_data, dtype=np.uint8).reshape((frame_width, frame_height, 3))
    topic = topic.decode('utf-8')
    timestamp = float(timestamp.decode('utf-8'))
    return topic, image, timestamp


def create_jpg_image_message(image, topic, timestamp=None, quality=100):
    if timestamp is None:
        timestamp = time.time()
    timestamp = str(timestamp).encode('utf-8')

    # Compress the image using JPEG compression with a quality factor
    success, compressed_image = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("Image compression failed")

    marshalled_message = [topic.encode('utf-8'), compressed_image.tobytes(), timestamp]
    return marshalled_message


def parse_jpg_image_message(message):
    topic, compressed_image_data, timestamp = message

    # Decompress the image
    compressed_image_array = np.frombuffer(compressed_image_data, dtype=np.uint8)
    image = cv2.imdecode(compressed_image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image decompression failed")

    topic = topic.decode('utf-8')
    timestamp = float(timestamp.decode('utf-8'))
    return topic, image, timestamp