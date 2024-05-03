import time
from contextlib import contextmanager

@contextmanager
def timer(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        elapsed_time = end - start
        #print(f"{label}: {format(elapsed_time, '.5f')} seconds")