from threading import Lock, Thread
from queue import Queue
import time

from ._functional import start_job
from ._sync import select_job


_JOB_PASS = Queue()


class JobSelection(Thread):
    def run(self):
        while True:
            job = select_job()

            while _JOB_PASS.qsize() > 10:
                time.sleep(0.05)

            _JOB_PASS.put(job)


class WorkerWatcher(Thread):
    def run(self):
        while True:
            num_working = 0
            for worker in workers:
                num_working += worker._running

            # print(f"{len(workers)=}, {num_working=}")

            if num_working == len(workers):
                worker = Worker()
                worker.setDaemon(True)
                worker.start()
                workers.append(worker)

            time.sleep(10.)


class Worker(Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._running = False

    def run(self):
        while True:
            job = _JOB_PASS.get()

            self._running = True

            # work
            start_job(job)

            self._running = False


job_selection = JobSelection()
job_selection.setDaemon(True)
job_selection.start()

workers = []
for _ in range(16):     # TODO; 16 as an arg
    worker = Worker()
    worker.setDaemon(True)
    worker.start()
    workers.append(worker)


worker_watcher = WorkerWatcher()
worker_watcher.setDaemon(True)
worker_watcher.start()
