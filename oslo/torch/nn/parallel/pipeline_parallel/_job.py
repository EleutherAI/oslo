from threading import Thread


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

    def requires_redirection(self):
        return self.is_training and self.is_grad_enabled


@total_ordering
class Job:
    def __init__(self, tensors, unique_key, args_stub, kwargs_stub, meta):
        self._tensors = tensors
        self._unique_key = unique_key
        self._args_stub = args_stub
        self._kwargs_stub = kwargs_stub
        self._meta = meta

    @property
    def tensors(self):
        return self._tensors

    @property
    def unique_key(self):
        return self._unique_key

    @property
    def args_stub(self):
        return self._args_stub

    @property
    def kwargs_stub(self):
        return self._kwargs_stub

    @property
    def meta(self):
        return self._meta

    def __lt__(self, other):
        other: Job
        return self.unique_key < other.unique_key

    def __eq__(self, other):
        other: Job
        return self.unique_key == other.unique_key
