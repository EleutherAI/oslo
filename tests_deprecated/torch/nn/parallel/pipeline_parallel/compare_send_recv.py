import os

import torch


def main():
    file_names = os.listdir("tmp/")
    send_names = [fn for fn in file_names if "send" in fn]

    for send_name in send_names:
        recv_name = send_name.replace("send", "recv")

        send_path = os.path.join("tmp", send_name)
        recv_path = os.path.join("tmp", recv_name)

        send_data = torch.load(send_path, map_location="cpu")
        recv_data = torch.load(recv_path, map_location="cpu")

        assert send_data["__KEY__"] == recv_data["__KEY__"]
        assert send_data["__META__"] == recv_data["__META__"]

        assert send_data["__VALUE__"]["stub"] == recv_data["__VALUE__"]["stub"]

        send_data = send_data["__VALUE__"]["tensors"]
        recv_data = recv_data["__VALUE__"]["tensors"]

        for x, y in zip(send_data, recv_data):
            assert torch.allclose(x, y, atol=1e-16), send_name
            assert x.dtype == y.dtype, send_name


if __name__ == "__main__":
    main()
