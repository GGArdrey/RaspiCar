import time
from contextlib import contextmanager
import logging

logger = logging.getLogger("Stopwatch")
logger.setLevel(logging.DEBUG)

@contextmanager
def timer(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        elapsed_time = end - start
        logger.debug(f"{label}: {format(elapsed_time, '.5f')} seconds")
