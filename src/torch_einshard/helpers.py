import os
import torch.distributed as dist

def init_process_group(backend, use_cuda):
    """init torch distributed process group
    """

    if 'SLURM_NTASKS' not in os.environ:
        world_size = int(os.getenv("WORLD_SIZE", 1))
        world_rank = int(os.getenv("RANK", 0))
    else:
        world_size = int(os.getenv("SLURM_NTASKS", 1))
        world_rank = int(os.getenv("SLURM_PROCID", 0))
    dist.init_process_group(backend=backend, rank=world_rank, world_size=world_size)

    if use_cuda:
        if 'LOCAL_RANK' in os.environ:
            local_rank = int(os.getenv("LOCAL_RANK"))
        else:
            return world_rank % torch.cuda.device_count()
    else:
        local_rank = 0

    return local_rank, world_rank, world_size

# TODO: add async_op option
def all_reduce(input, group):
    input = input.contiguous()
    dist.all_reduce(input, group = group)
    return input

