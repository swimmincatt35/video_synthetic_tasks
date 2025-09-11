import os
import time
import urllib.request
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

from torch.utils.data import Dataset, DataLoader, DistributedSampler
from abc import ABC, abstractmethod


CUDA_VISIBLE_DEVICES = int(os.environ["LOCAL_RANK"])

def setup_distributed():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank) 
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    return rank, world_size

def cleanup_distributed():
    dist.destroy_process_group()


class BaseImageSequenceDataset(Dataset, ABC):
    def __init__(self, rank, dataset_name, root="/ubc/cs/research/plai-scratch/chsu35/datasets", seed=None):
        """
        Base class for sequence datasets based on MNIST/CIFAR10.
        Handles loading, downloading, and caching images per class.
        """
        self.dataset_name = dataset_name.upper()
        self.seed = seed

        if self.dataset_name == "MNIST":
            dataset_dir = os.path.join(root, "MNIST")
            self.C, self.H, self.W = 1, 28, 28
            num_classes = 10
        elif self.dataset_name == "CIFAR10":
            dataset_dir = os.path.join(root, "CIFAR10")
            self.C, self.H, self.W = 3, 32, 32
            num_classes = 10
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        self.N = num_classes
        transform = transforms.Compose([transforms.ToTensor()])

        # ---- Download with rank 0, sync others ----
        if rank == 0:
            os.makedirs(dataset_dir, exist_ok=True)
            self.base_dataset = self._get_dataset(dataset_dir, transform, download=True)
        dist.barrier(device_ids=[rank])
        if rank != 0:
            self.base_dataset = self._get_dataset(dataset_dir, transform, download=False)

        # Cache all images per class
        self.class_images = [
            torch.stack([img for img, label in self.base_dataset if label == k])
            for k in range(self.N)
        ]

        # RNG
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def _get_dataset(self, dataset_dir, transform, download):
        """Helper to fetch MNIST or CIFAR10 dataset."""
        if self.dataset_name == "MNIST":
            return torchvision.datasets.MNIST(
                root=os.path.join(dataset_dir, "train"),
                train=True,
                transform=transform,
                download=download
            )
        elif self.dataset_name == "CIFAR10":
            return torchvision.datasets.CIFAR10(
                root=os.path.join(dataset_dir, "train"),
                train=True,
                transform=transform,
                download=download
            )

    def __len__(self):
        return 100_000  # arbitrary large size, should rely on sampler

    @abstractmethod
    def __getitem__(self, idx):
        """Must be implemented by subclasses."""
        pass


class InductionHeadDataset(BaseImageSequenceDataset):
    def __init__(self, rank, dataset_name="MNIST", root="/ubc/cs/research/plai-scratch/chsu35/datasets", L=256, seed=None):
        super().__init__(rank, dataset_name, root, seed)
        self.L = L
        self.special_token = torch.ones(self.C, self.H, self.W)

    def __getitem__(self, idx):
        L, N = self.L, self.N

        # Random class sequence
        cls_indices = torch.randint(0, N, (L,), generator=self.rng)  # [L]
        img_indices = torch.tensor([
            torch.randint(0, len(self.class_images[c]), (1,), generator=self.rng).item()
            for c in cls_indices
        ])

        seq = torch.zeros(L, self.C, self.H, self.W)
        labels = cls_indices.clone()

        # Fill sequence with images
        for i in range(N):
            mask = (cls_indices == i)
            if mask.any():
                seq[mask] = torch.stack([self.class_images[i][img_idx] for img_idx in img_indices[mask]])

        # Special token positions
        induct_head = torch.randint(0, L - 2, (1,), generator=self.rng).item()
        seq[induct_head] = self.special_token
        seq[L - 1] = self.special_token
        seq_label = labels[induct_head + 1].clone()

        return seq, seq_label



class SelectiveCopyDataset(BaseImageSequenceDataset):
    def __init__(self, rank, dataset_name="MNIST", root="/ubc/cs/research/plai-scratch/chsu35/datasets", L=4096, seed=42):
        super().__init__(rank, dataset_name, root, seed)
        self.L = L

    def __getitem__(self, idx):
        L, N, C, H, W = self.L, self.N, self.C, self.H, self.W

        # Sample unique positions for each class
        positions = torch.multinomial(torch.ones(L), N, replacement=False, generator=self.rng)
        _, sort_idx = torch.sort(positions)
        labels = torch.arange(N).gather(0, sort_idx)

        # Sample one image per class
        batch_indices = torch.tensor([
            torch.randint(0, self.class_images[k].shape[0], (1,), generator=self.rng).item()
            for k in range(N)
        ])
        batch_digits = torch.stack([
            self.class_images[k][batch_indices[k]] for k in range(N)
        ])

        video = torch.zeros(L, C, H, W)
        video[positions] = batch_digits
        return video, labels


def main(rank, seed=42):
    dataset_name = "CIFAR10"

    # Distributed Dataset

    #dataset = SelectiveCopyDataset(rank=rank, dataset_name=dataset_name, L=4096, seed=seed+rank)
    dataset = InductionHeadDataset(rank=rank, dataset_name=dataset_name, L=256, seed=seed+rank)
    sampler = DistributedSampler(dataset)  # handles multi-GPU split
    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=0, pin_memory=True) # num_workers might cause deadlock

    for batch_idx, (videos, labels) in enumerate(dataloader):
        print(f"[Rank {rank}] videos.shape = {videos.shape}, labels.shape = {labels.shape}, labels = {labels}")
        break

if __name__ == "__main__":
    seed=42
    rank, world_size = setup_distributed()
    main(rank,seed)
    cleanup_distributed()
