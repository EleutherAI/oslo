from threading import Lock, Thread
from queue import Queue

from ._functional import start_job
from ._sync import select_job


_JOBS = Queue()


class JobSelection(Thread):
    def run(self):
        while True:
            job = select_job()

            _JOBS.put(job)


class Worker(Thread):
    def run(self):
        while True:
            job = _JOBS.get()

            # work
            start_job(job)


job_selection = JobSelection()
job_selection.setDaemon(True)
job_selection.start()

workers = []
for _ in range(16):     # TODO; 16 as an arg
    worker = Worker()
    worker.setDaemon(True)
    worker.start()
    workers.append(worker)
