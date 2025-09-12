#!/usr/bin/env python3
"""
Training & validation template for RecurrentEncoder with distributed training + wandb.
"""
import argparse
import wandb
import os
import sys

# Add 'src' to Python path
src_path = os.path.join(os.path.dirname(__file__), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import DataLoader, DistributedSampler
from models.encoders import RecurrentEncoder
from seq_dataset import InductionHeadDataset, SelectiveCopyDataset 




def setup_distributed():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    return rank, world_size


def cleanup_distributed():
    dist.destroy_process_group()


def train_step(args, model, optimizer, videos, label, device):
    model.train()
    videos, labels = videos.to(device), labels.to(device) 

    optimizer.zero_grad()
    initial_states = model.get_initial_recurrent_state(videos.shape[0], device)
    logits = model(videos, initial_states)  #  [B,rollout,n_classes]
    logits = logits.reshape(-1, logits.size(-1))  # [B*rollout,n_classes]

    if args.synth_task=='sel_copy':
        labels = labels.reshape(-1)  # [B*rollout]

    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def evaluate(args, model, dataloader, criterion, device):
    model.eval()
    total_loss, logit_correct, total_logits = 0.0, 0, 0

    if args.synth_task:
        batch_correct = 0
        total_batches = 0

    for step, (videos, labels) in enumerate(dataloader):
        if step * args.batch_size >= args.eval_samples:
            break

        videos, labels = videos.to(device), labels.to(device)
        initial_states = model.get_initial_recurrent_state(videos.shape[0], device)
        logits = model(videos, initial_states)

        if args.synth_task=='sel_copy':
            labels = labels.reshape(-1)  # [B*rollout]

        loss = criterion(logits, labels)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=-1)
        logit_correct += (preds == labels).sum().item()
        total_logits += labels.numel()

        if args.synth_task=='sel_copy':
            # Per-sample correctness: only count as correct if *all* predictions are correct
            batch_result = (preds == labels).reshape(videos.shape[0], -1).all(dim=1)
            batch_correct += batch_result.sum().item()
            total_batches += videos.shape[0]
    
    # Convert to tensors for reduction
    total_loss_tensor = torch.tensor(total_loss, device=device)
    logit_correct_tensor = torch.tensor(logit_correct, device=device)
    total_logits_tensor = torch.tensor(total_logits, device=device)
    batch_correct_tensor = torch.tensor(batch_correct, device=device)
    total_batches_tensor = torch.tensor(total_batches, device=device)
    if dist.is_initialized():
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(logit_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_logits_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(batch_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_batches_tensor, op=dist.ReduceOp.SUM)
    
    avg_loss = total_loss_tensor.item() / (step + 1)
    accuracy = logit_correct_tensor.item() / total_logits_tensor.item()
    batch_accuracy = (
        batch_correct_tensor.item() / total_batches_tensor.item()
        if total_batches_tensor.item() > 0 else None
    )

    return avg_loss, accuracy, batch_accuracy


# --------------------------
# Main training routine
# --------------------------
def main(args):
    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}")

    # W&B init (only on rank 0)
    if rank == 0:
        wandb.init(project=args.project, config=vars(args), name=args.run_name)

    # Dataset + Dataloader
    if args.synth_task == 'ind_head':
        print(f"[Rank {rank}] ind_head")
        dataset = InductionHeadDataset(rank=rank, dataset_name=args.dataset_name, L=args.seq_length, seed=args.seed+rank)
        print(f"[Rank {rank}] ind_head done")
        
    elif args.synth_task.upper() == 'sel_copy':
        print(f"[Rank {rank}] sel_copy")
        dataset = SelectiveCopyDataset(rank=rank, dataset_name=args.dataset_name, L=args.seq_length, seed=args.seed+rank)
        print(f"[Rank {rank}] sel_copy done")

    else:
        raise ValueError(f"Unknown synthetic task {args.synth_task}")
    
    print(f"[Rank {rank}] hi1")
    sampler = DistributedSampler(dataset)

    print(f"[Rank {rank}] hi2")
    assert args.batch_size%world_size==0
    dataloader = DataLoader(dataset, batch_size=args.batch_size//world_size, sampler=sampler, num_workers=2, pin_memory=True)
    print(f"[Rank {rank}] hi3")

    # Model, optimizer, criterion
    model = RecurrentEncoder(
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        rnn_type=args.rnn_type,
        is_video_synth_task=True,
        video_synth_task_out_dim=10
    ).to(device)

    print(f"[Rank {rank}] hi4")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    if rank == 0:
        running_loss = 0.0

    for step, (videos, labels) in enumerate(dataloader):
        if step >= args.train_iters:
            break
        
        videos = videos.reshape(videos.size(0), videos.size(1), -1) # flatten
        
        print(f"[Rank {rank}] hi5")
        loss = train_step(args, model, optimizer, videos, label, device)
        print(f"[Rank {rank}] hi6")
        if rank == 0:
            running_loss += loss.item()
            running_loss /= (step+1)

        if (step+1) % 100 == 0 and rank == 0:
            print(f"[Step {step+1}] Running train Loss: {running_loss:.4f}")  
            if wandb_logger: 
                wandb_logger.log({"train/loss": loss.item()}, step=step+1)
        
        if (step+1) % args.eval_every == 0:
            eval_loss, eval_acc, eval_batch_acc = evaluate(args, model, dataloader, criterion, device)
            if rank == 0:
                print(f"[Step {step+1}] Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.4f}, Eval Batch Acc: {eval_batch_acc:.4f}")
                wandb.log({"eval/loss": eval_loss, "eval/acc": eval_acc, "eval/batch_acc": eval_batch_acc}, step=step+1)

        if (step+1) % args.save_every == 0 and rank == 0:
            ckpt_path = os.path.join(args.save_dir, f"checkpoint_epoch{step+1}.pt")
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.module.state_dict(), ckpt_path)
            print(f"[Rank 0] Saved checkpoint to {ckpt_path}")

    cleanup_distributed()


# --------------------------
# Entry point
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Training Template with W&B")

    # Model hyperparams
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--rnn_type", type=str, default="mingru")

    # Dataset
    parser.add_argument("--synth_task", type=str, default="ind_head", help="ind_head | sel_copy")
    parser.add_argument("--dataset_name", type=str, default="CIFAR10", help="MNIST | CIFAR10")
    parser.add_argument("--seq_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_iters", type=int, default=100, help="steps per epoch")
    parser.add_argument("--eval_samples", type=int, default=40)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")

    # W&B
    parser.add_argument("--project", type=str, default="minGRU-toy-run")
    parser.add_argument("--run_name", type=str, default="experiment-1")

    args = parser.parse_args()
    main(args)
