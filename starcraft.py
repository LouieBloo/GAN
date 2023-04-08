import socket
import sys
import json

HOST = 'localhost'
PORT = 8000


try: 
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
                # note this could be a problem in the future
                data = conn.recv(1024*1024)
                if not data:
                    break

                print('Received message:', data.decode())
                sys.stdout.flush()

                # Send a response back to the C++ app
                message = {'response': 'Hello from Python'}
                conn.sendall((json.dumps(message) + "\n").encode())
except socket.error as e:
    print(f"Socket error: {e}")