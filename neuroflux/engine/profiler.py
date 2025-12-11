import torch
import time
from contextlib import contextmanager

class NeuroProfiler:
    """
    High-precision CUDA event profiler for deep learning workloads.
    """
    def __init__(self):
        self.records = {}
        self.active_events = {}

    @contextmanager
    def record(self, name):
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            torch.cuda.synchronize()
            self.records[name] = start.elapsed_time(end)
        else:
            start = time.time()
            yield
            self.records[name] = (time.time() - start) * 1000

    def summary(self):
        print("\n=== NeuroFlux Kernel Profiler ===")
        print(f"{'Kernel Name':<30} | {'Duration (ms)':<15}")
        print("-" * 50)
        for name, duration in self.records.items():
            print(f"{name:<30} | {duration:.4f}")
        print("-" * 50)
