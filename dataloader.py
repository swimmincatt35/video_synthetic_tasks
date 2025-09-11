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

CUDA_VISIBLE_DEVICES = int(os.environ["LOCAL_RANK"])

def setup_distributed():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank) 
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    return rank, world_size

def cleanup_distributed():
    dist.destroy_process_group()


class InductionHeadDataset(Dataset):
    def __init__(self, rank, dataset_name="MNIST", root="/ubc/cs/research/plai-scratch/chsu35/datasets", L=256, seed=None):
        """
        dataset_name: "MNIST" or "CIFAR10"
        L: sequence length
        seed: random seed for reproducibility
        """
        self.dataset_name = dataset_name.upper()
        self.L = L
        self.seed = seed

        if dataset_name == "MNIST":
            dataset_dir = os.path.join(root, "MNIST")
            self.C, self.H, self.W = 1, 28, 28
            num_classes = 10
        elif dataset_name == "CIFAR10":
            dataset_dir = os.path.join(root, "CIFAR10")
            self.C, self.H, self.W = 3, 32, 32
            num_classes = 10
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        self.N = num_classes

        transform = transforms.Compose([transforms.ToTensor()])   # Converts to [0,1] float tensor
        os.makedirs(dataset_dir, exist_ok=True)

        # Allow rank 0 to download
        if rank == 0:
            if dataset_name=="MNIST":
                base_dataset = torchvision.datasets.MNIST(
                    root=os.path.join(dataset_dir, "train"),
                    train=True,
                    transform=transform,
                    download=True
                )
            elif dataset_name == "CIFAR10":
                base_dataset = torchvision.datasets.CIFAR10(
                    root=os.path.join(dataset_dir, "train"),
                    train=True,
                    transform=transform,
                    download=True
                )
        dist.barrier(device_ids=[rank])

        # Other ranks just reload from disk
        if rank != 0:
            if dataset_name.upper() == "MNIST":
                base_dataset = torchvision.datasets.MNIST(
                    root=os.path.join(dataset_dir, "train"),
                    train=True,
                    transform=transform,
                    download=False
                )
            elif dataset_name.upper() == "CIFAR10":
                base_dataset = torchvision.datasets.CIFAR10(
                    root=os.path.join(dataset_dir, "train"),
                    train=True,
                    transform=transform,
                    download=False
                )

        # Cache images per class
        self.class_images = [
            torch.stack([img for img, label in base_dataset if label == k])
            for k in range(self.N)
        ]

        # Special token: white image
        self.special_token = torch.ones(self.C, self.H, self.W)

        # RNG
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def __len__(self):
        return 100_000  # arbitrary large size

    def __getitem__(self, idx):
        L, N = self.L, self.N

        # --- Random positions for images ---
        cls_indices = torch.randint(0, N, (L,), generator=self.rng)  # [L] class labels
        img_indices = torch.tensor([
            torch.randint(0, len(self.class_images[c]), (1,), generator=self.rng).item()
            for c in cls_indices
        ]) # [L] sample random indices for diff classes 

        # --- Allocate tensors ---
        seq = torch.zeros(L, self.C, self.H, self.W)
        labels = cls_indices.clone()

        # --- Assign images ---
        for i in range(N):
            mask = (cls_indices == i)
            if mask.any():
                seq[mask] = torch.stack([self.class_images[i][img_idx] for img_idx in img_indices[mask]])

        # --- Choose first special token position ---
        induct_head = torch.randint(0, L-2, (1,), generator=self.rng).item()
        seq[induct_head] = self.special_token
        seq[L-1] = self.special_token
        seq_label = labels[induct_head + 1].clone()

        return seq, seq_label


class SelectiveCopyDataset(Dataset):
    def __init__(self, rank, dataset_name="MNIST", root="/ubc/cs/research/plai-scratch/chsu35/datasets", L=4096, seed=42):
        """
        dataset_name: str, e.g. "MNIST" or "CIFAR10"
        root: path to store dataset
        L: length of the video sequence
        N: number of distinct classes to insert (defaults to full dataset size)
        seed: optional random seed for reproducibility
        """
        self.dataset_name = dataset_name.upper()
        self.L = L
        self.seed = seed

        if dataset_name == "MNIST":
            dataset_dir = os.path.join(root, "MNIST")
            self.C, self.H, self.W = 1, 28, 28
            num_classes = 10
        elif dataset_name == "CIFAR10":
            dataset_dir = os.path.join(root, "CIFAR10")
            self.C, self.H, self.W = 3, 32, 32
            num_classes = 10
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        self.N = num_classes

        transform = transforms.Compose([transforms.ToTensor()])   # Converts to [0,1] float tensor
        os.makedirs(dataset_dir, exist_ok=True)

        # Allow rank 0 to download
        if rank == 0:
            if dataset_name=="MNIST":
                base_dataset = torchvision.datasets.MNIST(
                    root=os.path.join(dataset_dir, "train"),
                    train=True,
                    transform=transform,
                    download=True
                )
            elif dataset_name == "CIFAR10":
                base_dataset = torchvision.datasets.CIFAR10(
                    root=os.path.join(dataset_dir, "train"),
                    train=True,
                    transform=transform,
                    download=True
                )
        dist.barrier(device_ids=[rank])

        # Other ranks just reload from disk
        if rank != 0:
            if dataset_name.upper() == "MNIST":
                base_dataset = torchvision.datasets.MNIST(
                    root=os.path.join(dataset_dir, "train"),
                    train=True,
                    transform=transform,
                    download=False
                )
            elif dataset_name.upper() == "CIFAR10":
                base_dataset = torchvision.datasets.CIFAR10(
                    root=os.path.join(dataset_dir, "train"),
                    train=True,
                    transform=transform,
                    download=False
                )

        # Preload dataset per class
        self.all_images = [
            torch.stack([img for img, target in base_dataset if target == k])
            for k in range(self.N)
        ]

        # RNG
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def __len__(self):
        return 100_000  # arbitrary large size, use sampler in training

    def __getitem__(self, idx):
        L, N, C, H, W = self.L, self.N, self.C, self.H, self.W

        # Sample unique positions
        positions = torch.multinomial(torch.ones(L), N, replacement=False, generator=self.rng)
        _, sort_idx = torch.sort(positions)
        labels = torch.arange(N).gather(0, sort_idx)  # [N]

        # Sample digits per class
        batch_indices = torch.tensor([
            torch.randint(0, self.all_images[k].shape[0], (1,), generator=self.rng).item()
            for k in range(N)
        ])
        batch_digits = torch.stack([
            self.all_images[k][batch_indices[k]] for k in range(N)
        ])  # [N, C, H, W]

        # Scatter into dense video
        video = torch.zeros(L, C, H, W)
        video[positions] = batch_digits

        return video, labels


def main(rank, seed=42):
    dataset_name = "CIFAR10"

    # Distributed Dataset
    t0 = time.time()
    #dataset = SelectiveCopyDataset(rank=rank, dataset_name=dataset_name, L=4096, seed=seed+rank)
    dataset = InductionHeadDataset(rank=rank, dataset_name=dataset_name, L=256, seed=seed+rank)
    t1 = time.time()
    t0 = time.time()
    sampler = DistributedSampler(dataset)  # handles multi-GPU split
    t1 = time.time()
    t0 = time.time()
    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=0, pin_memory=True)
    t1 = time.time()


    for batch_idx, (videos, labels) in enumerate(dataloader):
        print(f"[Rank {rank}] videos.shape = {videos.shape}, labels.shape = {labels.shape}, labels = {labels}")
        break
    

if __name__ == "__main__":
    seed=42
    rank, world_size = setup_distributed()
    main(rank,seed)
    cleanup_distributed()
