#!/usr/bin/env python3
"""
Wrapper for NeMo oomptimizer that fixes NCCL segfault on single GPU.

The oomptimizer unconditionally initializes NCCL even on single GPU,
which causes segfaults with certain model configurations (e.g., Conformer).
This wrapper skips distributed init for single GPU since it's not needed
for batch size profiling.
"""
import sys
import runpy

import torch

original_init_pg = torch.distributed.init_process_group

def patched_init_pg(*args, **kwargs):
    if torch.cuda.device_count() == 1:
        return None  # Skip - not needed for single GPU profiling
    return original_init_pg(*args, **kwargs)

torch.distributed.init_process_group = patched_init_pg

if __name__ == "__main__":
    sys.argv[0] = "/opt/NeMo/scripts/speechlm2/oomptimizer.py"
    runpy.run_path("/opt/NeMo/scripts/speechlm2/oomptimizer.py", run_name="__main__")
