import time

from torch.distributed import rpc

from oslo.torch.distributed.parallel_mode import ParallelMode

# for watching whether every backward work is done or not
_JOBS_REQUIRE_BACKWARD = set()


def register_job_requires_backward(job_name):
    _JOBS_REQUIRE_BACKWARD.add(job_name)


def notify_backward_job_done(job_name):
    _JOBS_REQUIRE_BACKWARD.remove(job_name)


def get_num_jobs_require_backward_remaining():
    return len(_JOBS_REQUIRE_BACKWARD)


# for unique tag generation
_NUM_FORWARD_USED_COUNTER = dict()


def register_location_for_forward_counter(location):
    _NUM_FORWARD_USED_COUNTER[location] = 0


def make_unique_key(location, rank):
    cnt = _NUM_FORWARD_USED_COUNTER[location]
    unique_key = (location, cnt, rank)
    _NUM_FORWARD_USED_COUNTER[location] += 1
    return unique_key


def reset_forward_used_counter():
    for k in _NUM_FORWARD_USED_COUNTER:
        _NUM_FORWARD_USED_COUNTER[k] = 0


# dictionary for result broadcast
_RESULT_DICT = dict()

_RESULT_RECEIVED_MARKER = dict()


def set_result(ind, result):
    _RESULT_DICT[ind] = result
    _RESULT_RECEIVED_MARKER[ind] = True


def get_result(ind):
    while ind not in _RESULT_RECEIVED_MARKER:
        time.sleep(0.0)
    return _RESULT_DICT[ind]


def reset_result():
    _RESULT_DICT.clear()
    _RESULT_RECEIVED_MARKER.clear()


#
_CHECKER_BATCH_JOB_FINISHED = 0


def notify_batch_job_finished():
    global _CHECKER_BATCH_JOB_FINISHED
    _CHECKER_BATCH_JOB_FINISHED += 1


def wait_other_ranks(rank, context):
    global _CHECKER_BATCH_JOB_FINISHED

    # TODO; check the reason why we need this code block
    #  for checking batch job done.
    #  gradient computation goes wrong without this code
    for other in context.get_ranks_in_group(ParallelMode.PIPELINE):
        if other == rank:
            notify_batch_job_finished()
        else:
            rpc_dst = context.get_pipeline_rpc_worker_name(other)
            rpc.rpc_sync(
                to=rpc_dst,
                func=notify_batch_job_finished,
            )

    while _CHECKER_BATCH_JOB_FINISHED < context.get_world_size(ParallelMode.PIPELINE):
        time.sleep(0.0)

    # every ranks done; reset
    _CHECKER_BATCH_JOB_FINISHED = 0

    # wait for all backward pass execution
    while get_num_jobs_require_backward_remaining() != 0:
        time.sleep(0.0)
