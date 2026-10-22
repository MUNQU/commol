"""How many cores a job keeps busy."""

import time
from collections.abc import Callable


def cores(job: Callable[[], object]) -> float:
    """The mean number of cores `job` keeps busy: its CPU time over its wall time."""
    cpu, wall = time.process_time(), time.perf_counter()
    job()
    return (time.process_time() - cpu) / (time.perf_counter() - wall)
