import logging
import threading
import queue
import sys

class StreamlitLogHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        msg = self.format(record)
        self.log_queue.put(msg)

def setup_logging_to_queue(log_queue):
    handler = StreamlitLogHandler(log_queue)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return handler

class StdoutRedirector:
    def __init__(self, log_queue):
        self.log_queue = log_queue

    def write(self, string):
        if string.strip():
            self.log_queue.put(string.strip())

    def flush(self):
        pass
