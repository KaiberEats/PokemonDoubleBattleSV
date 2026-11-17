"""Low-overhead wall/thread-CPU timings across the poke-env and main threads."""
from contextlib import contextmanager
from functools import wraps
from threading import Lock
from time import perf_counter, thread_time
import os


class PerformanceTimings:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self._lock = Lock()
        self._rows = {}

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        wall, cpu = perf_counter(), thread_time()
        try:
            yield
        finally:
            elapsed, used_cpu = perf_counter() - wall, thread_time() - cpu
            with self._lock:
                row = self._rows.setdefault(name, [0, 0.0, 0.0, 0.0])
                row[0] += 1
                row[1] += elapsed
                row[2] += used_cpu
                row[3] = max(row[3], elapsed)

    def timed(self, name):
        def decorate(function):
            @wraps(function)
            def wrapped(*args, **kwargs):
                with self.measure(name):
                    return function(*args, **kwargs)
            return wrapped
        return decorate

    def snapshot(self, reset=False):
        with self._lock:
            rows = {key: tuple(value) for key, value in self._rows.items()}
            if reset:
                self._rows.clear()
        return rows

    def report(self):
        if not self.enabled:
            return
        rows = self.snapshot(reset=True)
        if not rows:
            return
        print('[TIMING] interval since previous report; wall totals may overlap', flush=True)
        print('[TIMING] name                    calls    wall_s thread_cpu_s   avg_ms   max_ms', flush=True)
        for name, (count, wall, cpu, maximum) in sorted(rows.items()):
            print(f'[TIMING] {name:24s} {count:6d} {wall:9.3f} {cpu:12.3f} '
                  f'{1000 * wall / count:8.3f} {1000 * maximum:8.3f}', flush=True)


TIMINGS = PerformanceTimings(os.environ.get('DOUBLEBATTLE_PROFILE', '1') != '0')


def synchronize_agent(agent):
    """Include queued GPU work in update timings; CPU path has no overhead."""
    if getattr(getattr(agent, 'device', None), 'type', None) == 'cuda':
        import torch
        torch.cuda.synchronize(agent.device)
