import os
import time

import urllib.request
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision
import torchvision.transforms as transforms

import torchvision.utils as vutils

# ===========================
# Base directory for all datasets
# ===========================
base_dir = "/ubc/cs/research/plai-scratch/chsu35/datasets"
os.makedirs(base_dir, exist_ok=True)

# Dataset-specific directories
mnist_dir = os.path.join(base_dir, "MNIST")
cifar_dir = os.path.join(base_dir, "CIFAR10")
dsprites_dir = os.path.join(base_dir, "dSprites")

os.makedirs(mnist_dir, exist_ok=True)
os.makedirs(cifar_dir, exist_ok=True)
os.makedirs(dsprites_dir, exist_ok=True)


class SelectiveCopyMNISTDataset(Dataset):
    def __init__(self, mnist_data, L=4096, N=10, seed=None):
        """
        mnist_data: torchvision.datasets.MNIST instance
        L: length of the video sequence
        N: number of MNIST digits to insert
        seed: optional random seed for reproducibility
        """
        self.mnist_data = mnist_data
        self.L = L
        self.N = N
        self.C, self.H, self.W = 1, 28, 28

        # Preload MNIST images per class in memory (cache)
        self.all_images = [
            mnist_data.data[mnist_data.targets==k].unsqueeze(1).float() / 255.0
            for k in range(N)
        ]

        # Optional RNG for reproducibility
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def __len__(self):
        # Arbitrary large number of samples; use DistributedSampler to split
        return 100_000

    def __getitem__(self, idx):
        B = 1  # Each call returns a single sequence
        L, N, C, H, W = self.L, self.N, self.C, self.H, self.W

        # --- Sample unique positions ---
        positions = torch.multinomial(torch.ones(L), N, replacement=False, generator=self.rng)
        positions, sort_idx = torch.sort(positions)
        labels = torch.arange(N).gather(0, sort_idx)

        # --- Sample digits by label ---
        batch_indices = torch.tensor([
            torch.randint(0, self.all_images[k].shape[0], (1,), generator=self.rng).item()
            for k in range(N)
        ])
        batch_digits = torch.stack([
            self.all_images[k][batch_indices[k]] for k in range(N)
        ])  # [N, C, H, W]

        # --- Scatter into dense video ---
        video = torch.zeros(L, C, H, W)
        video[positions] = batch_digits

        return video, labels

def main(rank, seed=42):
    # Transform for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),   # Converts to [0,1] float tensor
    ])

    # Load MNIST (preload in memory)
    mnist_data = torchvision.datasets.MNIST(root=os.path.join(mnist_dir, "train"), train=True, download=True, transform=transform)

    # Distributed Dataset
    dataset = SelectiveCopyMNISTDataset(mnist_data, L=4096, N=10, seed=seed+rank)
    sampler = DistributedSampler(dataset)  # handles multi-GPU split

    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=0, pin_memory=True)

    print(f"[Rank] {rank}, hello word")


    for batch_idx, (videos, labels) in enumerate(dataloader):
        print(f"[Rank {rank}] videos.shape = {videos.shape}, labels.shape = {labels.shape}, labels = {labels}")

        '''
        # videos: [B, L, C, H, W]
        B, L, C, H, W = videos.shape

        # Create a directory for saving frames
        save_dir = f"./video_frames_rank{rank}_batch{batch_idx}"
        os.makedirs(save_dir, exist_ok=True)

        # Loop over batch and video frames
        for b in range(B):
            video = videos[b]  # [L, C, H, W]
            for t in range(L):
                frame = video[t]  # [C, H, W]
                # Save frame as PNG
                filename = os.path.join(save_dir, f"video{b}_frame{t:04d}.png")
                vutils.save_image(frame, filename)

        print(f"[Rank {rank}] Saved video frames to {save_dir}")
        '''
        break

if __name__ == "__main__":
    # torchrun --nproc_per_node=2 dataloader.py
    seed=42

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    print(world_size)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    main(rank,seed)
    
    # your code here
    dist.destroy_process_group()
