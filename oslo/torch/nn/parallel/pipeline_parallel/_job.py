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


# TODO; make an abstract class
@total_ordering
class Job:
    def __init__(self, tensors, unique_key, stub, meta):
        self._tensors = tensors
        self._unique_key = unique_key
        self._stub = stub
        self._meta = meta

    # TODO; to use set, is this enough?
    def __hash__(self):
        return hash(self.unique_key)

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

    def __lt__(self, other):
        other: Job
        return self.unique_key < other.unique_key

    def __eq__(self, other):
        other: Job
        return self.unique_key == other.unique_key


@total_ordering
class JobInitialization:
    def __init__(self, fn, is_grad_enabled, unique_key, out_queue, **kwargs):
        self._fn = fn
        self._is_grad_enabled = is_grad_enabled
        self._unique_key = unique_key
        self._kwargs = kwargs
        self._out_queue = out_queue

    # TODO; to use set, is this enough?
    def __hash__(self):
        return hash(self.unique_key)

    @property
    def fn(self):
        return self._fn

    @property
    def is_grad_enabled(self):
        return self._is_grad_enabled

    @property
    def unique_key(self):
        return self._unique_key

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def out_queue(self):
        return self._out_queue

    def __lt__(self, other):
        other: JobInitialization
        return self.unique_key < other.unique_key

    def __eq__(self, other):
        other: JobInitialization
        return self.unique_key == other.unique_key


@total_ordering
class Handshake:
    def __init__(self, unique_key):
        self._unique_key = unique_key

    # TODO; to use set, is this enough?
    def __hash__(self):
        return hash(self.unique_key)

    @property
    def unique_key(self):
        return self._unique_key

    def __lt__(self, other):
        if isinstance(other, (JobInitialization, Job)):
            return True
        else:
            return self.unique_key < other.unique_key

    def __eq__(self, other):
        if isinstance(other, Handshake):
            return self.unique_key == other.unique_key
        else:
            return False
