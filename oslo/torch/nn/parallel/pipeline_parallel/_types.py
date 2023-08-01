from threading import RLock
from dataclasses import dataclass
from typing import Any


# dataclass that holds queues for synchronization
@dataclass
class SyncNotification:
    FORWARD_READY_NOTICE: Any = None
    FORWARD_START_NOTICE: Any = None
    FORWARD_FINISHED_NOTICE: Any = None
    LAST_BACKWARD_NOTICE: Any = None
    BATCH_FINISHED_NOTICE: Any = None
    OUTPUT: Any = None

    # initialization flag
    _initialized = False

    # to make rpc available
    def initialize(
        self,
        forward_ready_notice,
        forward_start_notice,
        forward_finished_notice,
        last_backward_notice,
        batch_finished_notice,
        out_queue,
    ):
        # TODO; print warning?
        if self._initialized:
            return

        self.FORWARD_READY_NOTICE = forward_ready_notice
        self.FORWARD_START_NOTICE = forward_start_notice
        self.FORWARD_FINISHED_NOTICE = forward_finished_notice
        self.LAST_BACKWARD_NOTICE = last_backward_notice
        self.BATCH_FINISHED_NOTICE = batch_finished_notice
        self.OUTPUT = out_queue

        self._initialized = True


# dataclass that holds queues for data transfer
@dataclass
class SyncQueues:
    # queues for inter communication
    RECV_QUEUES = dict()
    HANDSHAKE_QUEUES = dict()
    RESPONSE_QUEUES = dict()

    # queues for synchronize tensor group
    TENSOR_GROUP_SYNC_QUEUES = dict()
    TENSOR_GROUP_NOTIFICATION_QUEUES = dict()

    # TODO; proper location?
    JOBS = set()


# dataclass that holds information for communication
@dataclass
class CommunicationInformation:
    PARALLEL_CONTEXT: Any = None
    LOCK: Any = None

    # initialization flag
    _initialized = False

    # need to add parallel context to the function args
    # since parallel context is device local and cannot make to pickle
    def initialize(self, parallel_context):
        # TODO; print warning?
        if self._initialized:
            return

        self.PARALLEL_CONTEXT = parallel_context
        self.LOCK = RLock()

        self._initialized = True
