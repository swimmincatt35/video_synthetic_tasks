import os
import argparse
import urllib.request
import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from abc import ABC, abstractmethod


def infinite_dataloader(dataloader):
    """Yield batches forever."""
    while True:
        for batch in dataloader:
            yield batch


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
            print(f"[Rank {rank}] creating")
            os.makedirs(dataset_dir, exist_ok=True)
            self.base_dataset = self._get_dataset(dataset_dir, transform, download=True)
        print(f"[Rank {rank}] waiting")
        dist.barrier(device_ids=[rank])
        print(f"[Rank {rank}] passed")
        
        if rank != 0:
            print(f"[Rank {rank}] verifying")
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
        #return 2**31 - 1   # arbitrary large size.
        return 50

    @abstractmethod
    def __getitem__(self, idx):
        """Must be implemented by subclasses."""
        pass


class InductionHeadDataset(BaseImageSequenceDataset):
    def __init__(self, rank, dataset_name="MNIST", root="/ubc/cs/research/plai-scratch/chsu35/datasets", seq_len=256, seed=None):
        super().__init__(rank, dataset_name, root, seed)
        self.L = seq_len
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
    def __init__(self, rank, dataset_name="MNIST", root="/ubc/cs/research/plai-scratch/chsu35/datasets", seq_len=4096, seed=42):
        super().__init__(rank, dataset_name, root, seed)
        self.L = seq_len

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


if __name__ == "__main__":
    import time
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.utils.data import DataLoader, DistributedSampler

    CUDA_VISIBLE_DEVICES = int(os.environ["LOCAL_RANK"])

    def parse_args():
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description='Dataset training script')
        parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
        parser.add_argument('--dataset', type=str, default="MNIST", choices=["MNIST", "CIFAR10"], help='Dataset to use')
        parser.add_argument('--dataset_root', type=str, default="/ubc/cs/research/plai-scratch/chsu35/datasets", help='Root directory for datasets')
        parser.add_argument('--dataset_type', type=str, default="selective", choices=["selective", "induction"],
                             help='Type of dataset: selective (SelectiveCopyDataset) or induction (InductionHeadDataset)')
        parser.add_argument('--seq_len', type=int, default=-1, help='Sequence length for SelectiveCopyDataset')
        parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
        parser.add_argument('--num_steps', type=int, default=100, help='Number of training steps to run')
                        
        return parser.parse_args()

    def setup_distributed():
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(rank) 
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        return rank, world_size

    def cleanup_distributed():
        dist.destroy_process_group()

    def test_sequence_dataset_main(rank, args):
        # Create dataset based on type
        if args.dataset_type == "selective":
            dataset = SelectiveCopyDataset(
                rank=rank, 
                dataset_name=args.dataset, 
                seq_len=args.seq_len, 
                seed=args.seed + rank, 
                root=args.dataset_root
            )
        else:  # induction
            dataset = InductionHeadDataset(
                rank=rank, 
                dataset_name=args.dataset, 
                seq_len=args.seq_len,  
                seed=args.seed + rank, 
                root=args.dataset_root
            )
        
        inf_dataloader = infinite_dataloader(DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            num_workers=0, 
            pin_memory=True
        ))

        start = time.time()

        for i in range(args.num_steps):
            batch = next(inf_dataloader)
            videos, labels = batch
            print(f"[Rank {rank}] Step {i+1}/{args.num_steps} - videos.shape = {videos.shape}, labels.shape = {labels.shape}, labels = {labels}")
        
        end = time.time()
        print(f"[Rank {rank}] Time for {args.num_steps} steps: {end - start:.4f} seconds")
        print(f"[Rank {rank}] Avg step time: {(end - start) / args.num_steps:.6f} seconds")

    # Parse command line arguments
    args = parse_args()

    if args.seq_len==-1: # default
        if args.dataset_type=="selective":
            args.seq_len = 4096
        elif args.dataset_type=="induction":
            args.seq_len = 256
    
    print(f"Running with arguments:")
    print(f"  Seed: {args.seed}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Dataset type: {args.dataset_type}")
    print(f"  Dataset root: {args.dataset_root}")
    print(f"  Sequence length (seq_len): {args.seq_len}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of steps: {args.num_steps}")
    
    rank, world_size = setup_distributed()
    test_sequence_dataset_main(rank, args)
    cleanup_distributed()
