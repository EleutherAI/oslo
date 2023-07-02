from threading import Thread
from queue import Queue
import time

from ._functional import start_job
from ._sync import select_job, sleep


# a queue for communication between a job selector and workers
_JOBS_SELECTED = Queue()


class JobSelection(Thread):
    def run(self):
        while True:
            job = select_job()

            while _JOBS_SELECTED.qsize() > 10:
                sleep()

            _JOBS_SELECTED.put(job)


class WorkerPoolWatcher(Thread):
    def run(self):
        while True:
            num_working = 0

            for w in worker_pool:
                num_working += w.running

            # TODO; develop logic for controlling the number of workers
            if num_working == len(worker_pool) < 32:
                new_worker = Worker()
                new_worker.setDaemon(True)
                new_worker.start()
                worker_pool.append(new_worker)

            # sleep for enough time
            time.sleep(30.0)


class Worker(Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._running = False

    def run(self):
        while True:
            job = _JOBS_SELECTED.get()

            self._running = True

            # work
            start_job(job)

            self._running = False

    @property
    def running(self):
        return self._running


# worker for job selection
job_selection = JobSelection()
job_selection.setDaemon(True)
job_selection.start()


# workers run for forward and backward
worker_pool = []
for _ in range(16):  # TODO; 16 as an arg
    worker = Worker()
    worker.setDaemon(True)
    worker.start()
    worker_pool.append(worker)


# worker that controls the number of workers
pool_watcher = WorkerPoolWatcher()
pool_watcher.setDaemon(True)
pool_watcher.start()
