import websocket
import threading

websocket_url = "ws://localhost:5000"

def create_connection(endpoint, on_open=None, on_message=None, on_error=None, on_close=None):
    server_url = websocket_url + endpoint
    ws = websocket.WebSocketApp(server_url, on_open=on_open, on_message=on_message, on_close=on_close)
    thread = threading.Thread(target=ws.run_forever)
    thread.start()
    return ws

