import socket
import random


def get_free_port() -> int:
    """Get a free port on localhost.

    Returns:
        int: A free port on localhost.
    """
    while True:
        port = random.randrange(20000, 65000)
        try:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("localhost", port))
                return port
        except OSError:
            continue
