import time

from oslo.torch.nn.parallel.pipeline_parallel._job import Input
from oslo.torch.nn.parallel.pipeline_parallel._types import (
    SyncNotification,
    SyncQueues,
)


QUEUES = SyncQueues()

NOTIFICATIONS = SyncNotification()


def sleep():
    time.sleep(0.05)


def initialize_job(fn, is_grad_enabled, unique_key, out_queue, **kwargs):
    job = Input(
        fn=fn,
        is_grad_enabled=is_grad_enabled,
        unique_key=unique_key,
        out_queue=out_queue,
        **kwargs,
    )

    register_job(job)


def register_job(job):
    QUEUES.JOBS.add(job)


# TODO; support TP
def select_job():
    while len(QUEUES.JOBS) <= 0:
        sleep()

    job = list(sorted(QUEUES.JOBS))[0]
    QUEUES.JOBS.remove(job)
    return job


# for unique tag generation
_NUM_FORWARD_USED_COUNTER = dict()


def register_location_for_forward_counter(location):
    _NUM_FORWARD_USED_COUNTER[location] = 0


def make_unique_key(location, from_, to_):
    cnt = _NUM_FORWARD_USED_COUNTER[location]
    unique_key = (location, cnt, from_, to_)
    _NUM_FORWARD_USED_COUNTER[location] += 1
    return unique_key
