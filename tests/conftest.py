from types import SimpleNamespace
import os

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

import torch_einshard as es


def _use_cuda():
    return False


@pytest.fixture(scope="session")
def dist_env():
    use_cuda = _use_cuda()
    backend = "nccl" if use_cuda else "gloo"
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(20000 + os.getpid() % 10000))
    local_rank, world_rank, world_size = es.helpers.init_process_group(backend, use_cuda=use_cuda)

    if use_cuda:
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
        torch.cuda.manual_seed(333)
    else:
        device = "cpu"
    torch.manual_seed(333)

    yield SimpleNamespace(
        backend=backend,
        device=device,
        local_rank=local_rank,
        world_rank=world_rank,
        world_size=world_size,
    )

    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture(scope="session")
def mesh_1d(dist_env):
    return init_device_mesh(dist_env.device, (dist_env.world_size,), mesh_dim_names=("dp",))


@pytest.fixture(scope="session")
def mesh_2d(dist_env):
    if dist_env.world_size % 2 == 0:
        shape = (2, dist_env.world_size // 2)
    else:
        shape = (1, dist_env.world_size)
    return init_device_mesh(dist_env.device, shape, mesh_dim_names=("dp", "sp"))


def assert_close(actual, expected, *, rtol=1e-5, atol=1e-6):
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
