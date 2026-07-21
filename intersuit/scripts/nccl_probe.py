#!/usr/bin/env python
"""Small NCCL smoke test for single-node M4/AS-M4 training.

Run with torchrun. The probe intentionally avoids importing project modules so
it can distinguish distributed environment problems from model code problems.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tensor = torch.tensor([float(rank + 1)], device=torch.device("cuda", local_rank))
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    expected = world_size * (world_size + 1) / 2
    if rank == 0:
        print(
            {
                "world_size": world_size,
                "all_reduce_sum": float(tensor.item()),
                "expected": expected,
                "nccl_socket_ifname": os.environ.get("NCCL_SOCKET_IFNAME", ""),
            }
        )
    if abs(float(tensor.item()) - expected) > 1e-5:
        raise RuntimeError(f"NCCL all_reduce mismatch: got {float(tensor.item())}, expected {expected}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
