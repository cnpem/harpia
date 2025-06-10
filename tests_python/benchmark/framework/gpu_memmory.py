import GPUtil
from multiprocessing import Process, Event, Queue
import time

class MonitorProcess:
    def __init__(self, delay=0.5):
        self.delay = delay
        self.stop_event = Event()
        self.queue = Queue()
        self.process = Process(target=self._run, args=(self.stop_event, self.queue, delay))
        self.process.start()

    def _run(self, stop_event, queue, delay):
        memory_usage_log = []
        while not stop_event.is_set():
            gpus = GPUtil.getGPUs()
            mem_usage = [gpu.memoryUsed for gpu in gpus]
            memory_usage_log.append(mem_usage)
            time.sleep(delay)

        # When stopping, send back the full log
        queue.put(memory_usage_log)

    def stop(self):
        self.stop_event.set()
        self.process.join(timeout=2.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()

        memory_usage_log = []
        while not self.queue.empty():
            memory_usage_log = self.queue.get()

        if not memory_usage_log:
            return []

        return [
            max(mem[i] for mem in memory_usage_log)
            for i in range(len(memory_usage_log[0]))
        ]
