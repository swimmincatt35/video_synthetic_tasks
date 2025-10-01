#!/usr/bin/env python3
"""
Training & validation template for RecurrentEncoder with distributed training + wandb.
"""
import argparse
import wandb
import os
import sys
import json

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
from seq_dataset import InductionHeadDataset, SelectiveCopyDataset, infinite_dataloader 
from vae.vae import ConvVAE
import dist_utils


def setup_wandb(config: dict, wandb_config: dict, run_name: str = None, mode: str = None):
    """
    Initializes wandb with automatic offline fallback if internet is unavailable.
    """
    if mode is None:
        try:
            socket.create_connection(("api.wandb.ai", 443), timeout=3)
            mode = "online"
        except OSError:
            dist_utils.print0("[INFO] No internet detected: running wandb in offline mode.")
            mode = "offline"
    os.environ["WANDB_MODE"] = mode 
    wandb.login(key=wandb_config["api_key"])
    wandb.init(
        project=wandb_config["project"],
        entity=wandb_config["entity"],  
        name=run_name,
        mode=mode,  # redundant but explicit
        config=config
    )
    dist_utils.print0(f"[INFO] wandb initialized in {mode} mode. Run URL: {wandb.run.get_url() if wandb.run else 'N/A'}.")


def setup_distributed():
    """
    Initialize a distributed training environment using NCCL backend.
    Configures a custom global `print()` that prefixes messages with the process rank.
    """
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    dist_utils.setup_rank_print(rank)
    return rank, world_size


def cleanup_distributed():
    dist.destroy_process_group()




def train_step(args, model, latents, labels, criterion, device, grad_accum):
    model.train()
    initial_states = model.module.get_initial_recurrent_state(latents.shape[0], device)
    logits = model(latents, initial_states) 
    B, R, n_classes = logits.shape
    logits = logits.view(B * R, n_classes)           
    if args.synth_task == 'sel_copy':
        labels = labels.view(B * R)    
    loss = criterion(logits, labels)    
    (loss / grad_accum).backward()
    return loss.detach()   


@torch.no_grad()
def evaluate(args, model, inf_dataloader, criterion, vae, world_size, device, grad_accum):
    model.eval()
    total_loss, logit_correct, total_logits = 0.0, 0, 0    
    batch_correct = 0
    total_batches = 0

    step = 0
    batch_size = args.batch_size // grad_accum

    while step * batch_size < args.eval_samples:

        videos, labels = next(inf_dataloader) 
        videos, labels = videos.to(device), labels.to(device) 
        if args.synth_task == "ind_head":
            B, L, C, H, W = videos.shape 
            videos = videos.view(B*L, C, H, W)  
            with torch.no_grad():
                latents, _ = vae.encoder(videos)  
            latents = latents.view(B, L, -1)
        else: # sel_copy in latent_mode
            latents = videos

        initial_states = model.module.get_initial_recurrent_state(latents.shape[0], device)
        logits = model(latents, initial_states)
        B, R, n_classes = logits.shape
        logits = logits.view(B * R, n_classes)           
        if args.synth_task=='sel_copy':
            labels = labels.view(B * R)    

        loss = criterion(logits, labels)
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        logit_correct += (preds == labels).sum().item()
        total_logits += labels.numel()

        if args.synth_task=='sel_copy':
            # Per-sample correctness: only count as correct if *all* predictions are correct
            batch_result = (preds == labels).reshape(latents.shape[0], -1).all(dim=1)
            batch_correct += batch_result.sum().item()
            total_batches += latents.shape[0]
        
        step+=1

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
    
    avg_loss = total_loss_tensor.item() / (step * world_size)
    accuracy = logit_correct_tensor.item() / total_logits_tensor.item()
    batch_accuracy = (
        batch_correct_tensor.item() / total_batches_tensor.item()
        if total_batches_tensor.item() > 0 else 0
    )
    return avg_loss, accuracy, batch_accuracy


def load_vae_checkpoint(model, path, device):
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    else:
        raise FileNotFoundError(f"[ERROR] No checkpoint found at {path}")


# --------------------------
# Main training routine
# --------------------------
def main(args):
    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}")

    assert args.batch_size % world_size == 0
    global_batch_size = args.batch_size
    batch_size = int(args.batch_size // world_size)

    # Setup wandb
    if rank == 0:
        dist_utils.print0(f"Loading wandb config file: {args.wandb_config}")
        run_name = f"{args.synth_task}-{args.dataset_name}-{args.rnn_type}-ly{args.num_layers}-b{global_batch_size}-lr{args.lr}-{int(args.train_iters//1000)}k"
        if args.fixed_head > -1:
            run_name = f"fixed{args.fixed_head}-{run_name}"
        if args.seq_len > -1:
            run_name = f"seq{args.seq_len}-{run_name}"
        with open(args.wandb_config) as f:
            wandb_config = json.load(f)
        if args.tc:
            run_name = f"tc-s{args.stages}-p{args.b}-{run_name}"
        setup_wandb(config=vars(args), wandb_config=wandb_config, run_name=run_name, mode="offline")
    dist_utils.print0(f"Training configs: {vars(args)}")

    # Model+ddp
    latent_dim = 4*7*7 if args.dataset_name=="mnist" else 4*8*8
    synth_task_rollout_len = 10 if args.synth_task=="sel_copy" else 1
    model = RecurrentEncoder(
        output_dim=latent_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        rnn_type=args.rnn_type,
        is_video_synth_task=True,
        video_synth_task_out_dim=10,
        synth_task_rollout_len=synth_task_rollout_len
    ).to(device)
    dist_utils.print0("Setting up ddp... ")
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # VAE
    if args.dataset_name=="mnist":
        in_ch = out_ch = 1
    elif args.dataset_name=="cifar10":
        in_ch = out_ch = 3
    dist_utils.print0("Setting up vae... ")
    vae = ConvVAE(in_ch=in_ch, out_ch=out_ch, latent_ch=4, base_ch=32).to(device)
    load_vae_checkpoint(vae, args.vae_path, device)
    vae.eval() 

    # Dataset+Dataloader
    dist_utils.print0("Setting up dataset + dataloader... ")
    if args.synth_task == 'ind_head':
        seq_len = 256 if args.seq_len == -1 else args.seq_len
        dataset = InductionHeadDataset(rank=rank, dataset_name=args.dataset_name, seq_len=seq_len, 
                                       seed=args.seed+rank, root=args.dataset_root, fixed_head=args.fixed_head
                                       )
    elif args.synth_task == 'sel_copy':
        seq_len = 4096 if args.seq_len == -1 else args.seq_len
        dataset = SelectiveCopyDataset(rank=rank, dataset_name=args.dataset_name, seq_len=seq_len, 
                                       seed=args.seed+rank, root=args.dataset_root, use_latent=True, vae=vae)
    else:
        raise ValueError(f"Unknown synthetic task {args.synth_task}")
    inf_dataloader = infinite_dataloader(
        DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
    )

    # Optimizer
    optimizer = optim.AdamW(ddp.parameters(), lr=args.lr)
    optimizer.zero_grad()

    # Loss
    criterion = nn.CrossEntropyLoss()
    if rank == 0:
        running_loss = 0.0

    # Train
    grad_accum = 1
    dist_utils.print0("Training... ")
    if args.tc:
        dist_utils.print0(f"Training curriculum enabled, stages: {args.stages}, b: {args.b}.")
        assert args.train_iters % args.stages == 0
        iters = args.train_iters // args.stages
        stages = args.stages
    else:
        iters = args.train_iters
        stages = 1

    for stage in range(stages):
        if args.tc:
            dist_utils.print0(f"[Stage {stage}] Start.")

        for step in range(iters):  
            global_step = stage * iters + step

            loss = 0
            for accum in range(grad_accum):
                videos, labels = next(inf_dataloader) 
                videos, labels = videos.to(device), labels.to(device) 
                if args.synth_task == "ind_head":
                    B, L, C, H, W = videos.shape 
                    #print(videos.shape) # [B, L, 3, H, W]
                    #print(labels.shape) # [B] 
                    videos = videos.view(B*L, C, H, W)  # [B*L, 3, H, W]
                    with torch.no_grad():
                        latents, _ = vae.encoder(videos)  # [B*L, latent_dim]
                    latents = latents.view(B, L, -1)
                else:
                    assert dataset.is_latent_mode()
                    latents = videos
                loss += train_step(args, ddp, latents, labels, criterion, device, grad_accum)

            optimizer.step()
            optimizer.zero_grad()

            if rank == 0:
                running_loss = (running_loss * global_step + loss.item()) / (global_step+1)

            if (global_step+1) % args.log_every == 0:
                torch.cuda.synchronize()
                peak_allocated_gb = torch.cuda.max_memory_allocated(device) / 1024**3
                peak_reserved_gb = torch.cuda.max_memory_reserved(device) / 1024**3
                dist_utils.print0(f"[Step {global_step+1}] Peak Allocated: {peak_allocated_gb:.2f} GB | Peak Reserved: {peak_reserved_gb:.2f} GB")
                torch.cuda.reset_peak_memory_stats(device)
                if rank==0:
                    dist_utils.print0(f"[Step {global_step+1}] Running train Loss: {running_loss:.4f}") 
                    wandb.log({"train/loss": loss.item(),}, step=global_step+1) 
            
            if (global_step+1) % args.eval_every == 0:
                dist_utils.print0("Evaluating... ")
                eval_loss, eval_acc, eval_batch_acc = evaluate(args, ddp, inf_dataloader, criterion, vae, world_size, device, grad_accum)
                if rank == 0:
                    dist_utils.print0(f"[Step {global_step+1}] Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.4f}, Eval Batch Acc: {eval_batch_acc:.4f}")
                    wandb.log({"eval/loss": eval_loss, "eval/acc": eval_acc, "eval/batch_acc": eval_batch_acc}, step=global_step+1)
            
            if (global_step + 1) % args.save_every == 0 and rank == 0:
                save_dir = os.path.join(args.save_dir, run_name)
                os.makedirs(save_dir, exist_ok=True)
                # Save model
                ckpt_path = os.path.join(save_dir, f"ckpt_{global_step+1}.pt")
                torch.save(ddp.module.state_dict(), ckpt_path)
                dist_utils.print0(f"Saved model checkpoint to {ckpt_path}")
                # Save optimizer
                opt_path = os.path.join(save_dir, f"optimizer_{global_step+1}.pt")
                torch.save(optimizer.state_dict(), opt_path)
                dist_utils.print0(f"Saved optimizer state to {opt_path}")

        if args.tc:
            dist_utils.print0(f"[Stage {stage}] Done.")
            if stage != (stages-1):
                seq_len *= int(args.b)
                dist_utils.print0(f"[Stage {stage}] End of stage {stage}. Increase seq length to {seq_len}.")
                if args.synth_task == 'ind_head':
                    dataset = InductionHeadDataset(rank=rank, dataset_name=args.dataset_name, seq_len=seq_len, 
                                                   seed=args.seed+rank, root=args.dataset_root, fixed_head=args.fixed_head
                                                   )
                elif args.synth_task == 'sel_copy':
                    dataset = SelectiveCopyDataset(rank=rank, dataset_name=args.dataset_name, seq_len=seq_len, 
                                                   seed=args.seed+rank, root=args.dataset_root, use_latent=True, vae=vae)
                    batch_size = int(batch_size//int(args.b))
                    grad_accum = int(grad_accum*int(args.b))
                    dist_utils.print0(f"[Stage {stage}] End of stage {stage}. Increase grad accum to {grad_accum}. Decrease batch to {batch_size}.")
                inf_dataloader = infinite_dataloader(
                    DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
                )
    
    dist_utils.print0("Done...")
    cleanup_distributed()               


# --------------------------
# Entry point
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Training Template with W&B")

    # Model hyperparams
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--rnn_type", type=str, default="mingru")

    # VAE
    parser.add_argument("--vae_path", type=str, required=True, help="Path to vae.")

    # Dataset
    parser.add_argument("--dataset_root", type=str, default="/ubc/cs/research/plai-scratch/chsu35/datasets", help="Root directory for dataset, normally in scratch.")
    parser.add_argument("--dataset_name", type=str, default="cifar10", choices=["mnist", "cifar10"])
    parser.add_argument("--synth_task", type=str, default="ind_head", choices=["ind_head", "sel_copy"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    # Training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_iters", type=int, default=100, help="Steps per epoch.")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_samples", type=int, default=40)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="./rnn-runs")

    # Curriculum
    parser.add_argument("--tc", action="store_true", help="Enable curriculum.")
    parser.add_argument("--stages", type=int, default=4, help="Number of stages.")
    parser.add_argument("-b", type=int, default=4, help="Curriculum parameter.")

    # W&B
    parser.add_argument("--wandb_config", type=str, default=None)

    # Debug fixed_head, seq_len
    parser.add_argument("--fixed_head", type=int, default=-1)
    parser.add_argument("--seq_len", type=int, default=-1)

    args = parser.parse_args()
    main(args)
