"""
Helpers for distributed training.
"""

import io
import os
import socket
import functools

import blobfile as bf
import torch as th
import torch.distributed as dist
from mpi4py import MPI

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


# 检查环境变量来决定是否使用分布式训练
USE_DISTRIBUTED = os.environ.get("USE_DISTRIBUTED", "0") == "1"

# 为单机模式提供的辅助函数
def get_world_size():
    """在单机模式下返回世界大小为1"""
    if not USE_DISTRIBUTED:
        return 1
    return dist.get_world_size()

def get_rank():
    """在单机模式下返回rank为0"""
    if not USE_DISTRIBUTED:
        return 0
    return dist.get_rank()

# 替换原生的分布式函数
if not USE_DISTRIBUTED:
    # 保存原来的函数
    _original_get_world_size = dist.get_world_size
    _original_get_rank = dist.get_rank
    
    # 使用我们的函数替换
    dist.get_world_size = get_world_size
    dist.get_rank = get_rank

def setup_dist():
    """
    Setup a distributed process group.
    """
    # 如果设置为单机模式，则跳过分布式初始化
    if not USE_DISTRIBUTED:
        print("单机训练模式: 跳过分布式训练初始化")
        return
        
    if dist.is_initialized():
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"

    comm = MPI.COMM_WORLD
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    # 单机模式下直接加载文件
    if not USE_DISTRIBUTED:
        return th.load(path, **kwargs)
    
    # 分布式模式下的处理
    chunk_size = 2 ** 30  # MPI has a relatively small size limit
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    else:
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)

    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    # 如果是单机训练模式，跳过参数同步
    if not USE_DISTRIBUTED:
        return
        
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
