#!/usr/bin/env python3
"""
Testing script for RecurrentEncoder with saved checkpoints.
Loads model + optimizer (optional) and evaluates on given dataset.
"""

import argparse
import os
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

# Add 'src' to Python path
import sys
src_path = os.path.join(os.path.dirname(__file__), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from models.encoders import RecurrentEncoder
from seq_dataset import InductionHeadDataset, SelectiveCopyDataset, infinite_dataloader
from vae.vae import ConvVAE
import dist_utils
from train import setup_distributed, cleanup_distributed, evaluate, load_vae_checkpoint


def main(args):
    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}")

    dist_utils.print0(f"Testing args: {vars(args)}")

    # --------------------------
    # Load model & DDP
    # --------------------------
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
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    dist_utils.print0(f"Loading checkpoint: {args.rnn_ckpt_path}")
    checkpoint = torch.load(args.rnn_ckpt_path, map_location=device)
    ddp.module.load_state_dict(checkpoint)
    dist_utils.print0("Model checkpoint loaded successfully.")
    dist_utils.print0(f"Model was trained on sequence length: {args.train_seq_len}.")

    # --------------------------
    # Setup VAE
    # --------------------------
    if args.dataset_name=="mnist":
        in_ch = out_ch = 1
    elif args.dataset_name=="cifar10":
        in_ch = out_ch = 3
    vae = ConvVAE(in_ch=in_ch, out_ch=out_ch, latent_ch=4, base_ch=32).to(device)
    load_vae_checkpoint(vae, args.vae_path, device)
    vae.eval()
    dist_utils.print0("VAE loaded successfully.")

    # --------------------------
    # Setup Dataset + Dataloader
    # --------------------------
    for i in range(13,15): # [8,16384] [256,4096]
        seq_len = 2**i
        dist_utils.print0(f"====== ======")
        dist_utils.print0(f"Setting up dataset + dataloader of sequence length: {2**i}.")
        if args.synth_task == 'ind_head':
            dataset = InductionHeadDataset(rank=rank, dataset_name=args.dataset_name, seq_len=seq_len, 
                                           seed=args.seed+rank, root=args.dataset_root)
        elif args.synth_task == 'sel_copy':
            dataset = SelectiveCopyDataset(rank=rank, dataset_name=args.dataset_name, seq_len=seq_len,
                                           seed=args.seed+rank, root=args.dataset_root, use_latent=True, vae=vae)
        else:
            raise ValueError(f"Unknown synthetic task {args.synth_task}")
        inf_dataloader = infinite_dataloader(
            DataLoader(dataset, batch_size=args.batch_size//world_size, num_workers=0, pin_memory=True)
        )

        # Add a broadcast here
        dist.barrier(device_ids=[rank])

        # --------------------------
        # Loss & Evaluation
        # --------------------------
        criterion = nn.CrossEntropyLoss()
        dist_utils.print0(f"Running evaluation ...")
        eval_loss, eval_acc, eval_batch_acc = evaluate(args, ddp, inf_dataloader, criterion, vae, world_size, device)
        dist_utils.print0(f"[RESULT] Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.4f}, Eval Batch Acc: {eval_batch_acc:.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for trained model.")

    # Dataset
    parser.add_argument("--dataset_root", type=str, default="/ubc/cs/research/plai-scratch/chsu35/datasets")
    parser.add_argument("--dataset_name", type=str, default="cifar10", choices=["mnist", "cifar10"])
    parser.add_argument("--synth_task", type=str, default="ind_head", choices=["ind_head", "sel_copy"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    # VAE
    parser.add_argument("--vae_path", type=str, required=True, help="Path to VAE checkpoint.")

    # Model checkpoint and config
    parser.add_argument("--train_seq_len", type=int, required=True, help="Sequence length the RNN is trained on.")
    parser.add_argument("--rnn_ckpt_path", type=str, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--rnn_type", type=str, default="mingru")

    parser.add_argument("--eval_samples", type=int, default=40)

    args = parser.parse_args()
    main(args)

