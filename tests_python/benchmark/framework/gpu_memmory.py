import GPUtil
from multiprocessing import Process, Event, Queue
import time

class MonitorProcess:
    def __init__(self, delay=0.5):
        self.delay = delay
        self.stop_event = Event()
        self.queue = Queue()

        # Initial memory usage BEFORE subprocess starts
        gpus = GPUtil.getGPUs()
        self.initial_mem = [gpu.memoryUsed for gpu in gpus]

        # Start subprocess
        self.process = Process(target=self._run, args=(self.stop_event, self.queue, delay))
        self.process.start()

    def _run(self, stop_event, queue, delay):
        memory_usage_log = []

        while not stop_event.is_set():
            gpus = GPUtil.getGPUs()
            mem_usage = [gpu.memoryUsed for gpu in gpus]
            memory_usage_log.append(mem_usage)
            time.sleep(delay)

        queue.put(memory_usage_log)

    def stop(self, absolute_max=False):
        self.stop_event.set()
        self.process.join(timeout=2.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()

        # Final memory usage (AFTER subprocess ends)
        gpus = GPUtil.getGPUs()
        final_mem = [gpu.memoryUsed for gpu in gpus]

        memory_usage_log = []
        while not self.queue.empty():
            memory_usage_log = self.queue.get()

        if not memory_usage_log:
            return []

        max_mem = [max(mem[i] for mem in memory_usage_log) for i in range(len(memory_usage_log[0]))]

        if absolute_max:
            return max_mem

        return [x for i, m, f in zip(self.initial_mem, max_mem, final_mem) for x in (i, m, f)]
 