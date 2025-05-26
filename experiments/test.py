import socket


def check_buffer_sizes(self):
    try:
        sock = self.socket._sock
        recv_buf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        send_buf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        print(f"Receive buffer: {recv_buf} bytes ({recv_buf / 1024:.1f} KB)")
        print(f"Send buffer: {send_buf} bytes ({send_buf / 1024:.1f} KB)")
        return recv_buf, send_buf
    except Exception as e:
        print(f"Error checking buffer sizes: {e}")
