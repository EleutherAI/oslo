from dataclasses import dataclass
from functools import total_ordering


@dataclass
class Metadata:
    # message direction
    is_request: bool

    # job info
    is_forward: bool
    is_training: bool
    is_grad_enabled: bool
    is_fp16: bool
    func_name: str

    # communication info
    src: int
    dst: int


_ORDERING = dict()


@total_ordering
class AbstractJob:
    def __init__(self, *args, **kwargs):
        self._unique_key = ""

    @property
    def unique_key(self):
        return self._unique_key

    # TODO; to use set, is this enough?
    def __hash__(self):
        return hash(self.unique_key)

    def __lt__(self, other):
        if self.__class__ == other.__class__:
            if isinstance(self.unique_key, tuple):
                if self.unique_key[1] == other.unique_key[1]:
                    return self.unique_key < other.unique_key
                else:
                    return self.unique_key[1] < other.unique_key[1]
        else:
            return _ORDERING[self.__class__] < _ORDERING[other.__class__]

    def __eq__(self, other):
        return self.__class__ == other.__class__ and _ORDERING[self.__class__] == _ORDERING[other.__class__]


class Job(AbstractJob):
    def __init__(self, tensors, unique_key, stub, meta):
        super().__init__()

        self._tensors = tensors
        self._unique_key = unique_key
        self._stub = stub
        self._meta = meta

    @property
    def tensors(self):
        return self._tensors

    @property
    def unique_key(self):
        return self._unique_key

    @property
    def stub(self):
        return self._stub

    @property
    def meta(self):
        return self._meta


_ORDERING[Job] = 100


class Backward(Job):
    pass


_ORDERING[Backward] = 90


class Input(AbstractJob):
    def __init__(self, fn, is_grad_enabled, unique_key, out_queue, **kwargs):
        super().__init__()
        self._fn = fn
        self._is_grad_enabled = is_grad_enabled
        self._unique_key = unique_key
        self._kwargs = kwargs
        self._out_queue = out_queue

    @property
    def fn(self):
        return self._fn

    @property
    def is_grad_enabled(self):
        return self._is_grad_enabled

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def out_queue(self):
        return self._out_queue


_ORDERING[Input] = 50


class Handshake(AbstractJob):
    def __init__(self, src, dst, recv_key):
        super().__init__()
        self._src = src
        self._dst = dst
        self._unique_key = recv_key
        self._recv_key = recv_key

    @property
    def src(self):
        return self._src

    @property
    def dst(self):
        return self._dst

    @property
    def recv_key(self):
        return self._recv_key


class HandshakeRequest(Handshake):
    pass


_ORDERING[HandshakeRequest] = 1


class HandshakeResponse(Handshake):
    pass


_ORDERING[HandshakeResponse] = 0
