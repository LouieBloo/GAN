import socket
import sys
import json

HOST = 'localhost'
PORT = 8000

# Create a TCP socket and bind it to the specified host and port
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()

    with conn:
        print('Connected by', addr)
        sys.stdout.flush()
        while True:
            # Receive a message from the C++ app
            data = b''
            while True:
                chunk = conn.recv(1024)
                if not chunk:
                    break
                data += chunk

            if not data:
                break

            # Parse JSON data
            json_data = json.loads(data.decode())

            print('Received message:', json_data)
            sys.stdout.flush()

            # Send a response back to the C++ app
            message = {'response': 'Hello from Python'}
            conn.sendall((json.dumps(message) + "\n").encode())