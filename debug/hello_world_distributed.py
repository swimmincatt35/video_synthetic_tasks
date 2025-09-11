import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

CUDA_VISIBLE_DEVICES = int(os.environ["LOCAL_RANK"])

def setup_distributed():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank) 
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    return rank, world_size

def cleanup_distributed():
    dist.destroy_process_group()

def main(rank, seed=42):
    print(f"[Rank {rank}] Hello World")
    dist.barrier(device_ids=[rank])
    print(f"[Rank {rank}] Bye World")

if __name__ == "__main__":
    seed=42
    rank, world_size = setup_distributed()
    print(torch.cuda.current_device())
    main(rank,seed)
    cleanup_distributed()
