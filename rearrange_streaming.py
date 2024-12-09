import numpy as np
import socket
import time

def stream_data(file_path, ip, port, delay=1):
    data = np.load(file_path, allow_pickle=True)['data']

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    for row in data:
        message = ','.join(map(str, row))
        sock.sendto(message.encode(), (ip, port))
        print(f"Sent: {message}")
        time.sleep(delay)

if __name__ == "__main__":
    # Example usage with default IP and port
    stream_data('rearranged_position_temperature_data.npz', "127.0.0.1", 5005)
