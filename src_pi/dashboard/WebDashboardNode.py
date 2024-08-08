import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import zmq
import cv2
import base64
import numpy as np
import json
from threading import Thread

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

app.layout = html.Div([
    html.Div([
        html.Img(id='live-image', style={'width': '640px', 'height': '480px'}),
        dcc.Graph(id='live-graph', style={'width': '640px', 'height': '480px'}),
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    dcc.Interval(
        id='interval-component',
        interval=1 * 1000,  # in milliseconds
        n_intervals=0
    )
])

context = zmq.Context()
image_socket = context.socket(zmq.SUB)
image_socket.connect("tcp://localhost:5555")
image_socket.setsockopt_string(zmq.SUBSCRIBE, 'frame')

gamepad_socket = context.socket(zmq.SUB)
gamepad_socket.connect("tcp://localhost:5556")
gamepad_socket.setsockopt_string(zmq.SUBSCRIBE, 'gamepad')

image_data = None
gamepad_data = None


def receive_data():
    global image_data, gamepad_data
    poller = zmq.Poller()
    poller.register(image_socket, zmq.POLLIN)
    poller.register(gamepad_socket, zmq.POLLIN)

    while True:
        socks = dict(poller.poll())
        if image_socket in socks:
            topic, *msg_parts = image_socket.recv_multipart()
            if topic == b'frame':
                image_data = np.frombuffer(msg_parts[0], np.uint8)
        if gamepad_socket in socks:
            message = gamepad_socket.recv_string()
            topic, payload = message.split(' ', 1)  # Split the topic and the JSON payload
            controller_data = json.loads(payload)
            timestamp = controller_data["timestamp"]
            gamepad_data = controller_data["data"]


Thread(target=receive_data, daemon=True).start()


@app.callback(Output('live-image', 'src_pi'),
              [Input('interval-component', 'n_intervals')])
def update_image_src(n):
    global image_data
    if image_data is not None:
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        _, buffer = cv2.imencode('.jpg', image)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return 'data:image/jpeg;base64,' + jpg_as_text
    return None


@app.callback(Output('live-graph', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_graph(n):
    global gamepad_data
    if gamepad_data:
        throttle = gamepad_data['right_trigger']
        steering = gamepad_data['left_stick_y']
        return {
            'data': [
                go.Bar(
                    x=['Throttle', 'Steering'],
                    y=[throttle, steering],
                    marker=dict(color=['blue', 'red'])
                )
            ],
            'layout': go.Layout(
                yaxis=dict(range=[-1, 1])
            )
        }
    return {
        'data': [],
        'layout': go.Layout(
            yaxis=dict(range=[-1, 1])
        )
    }

if __name__ == '__main__':
    app.run_server(debug=True)
