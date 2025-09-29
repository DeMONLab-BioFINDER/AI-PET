# src/distrib.py
import os
import torch
import torch.distributed as dist

def ddp_init():
    """
    Initialize DDP if torchrun env variables are present.
    Returns (is_ddp, rank, world_size, local_rank).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return True, rank, world, local_rank
    return False, 0, 1, 0

def ddp_cleanup():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def is_rank0():
    return (not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0)
