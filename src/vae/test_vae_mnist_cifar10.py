import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import argparse, os, json
import wandb

from vae import ConvVAE 

@torch.no_grad()
def evaluate_reconstruction(model, dataloader, device, outdir, num_batches=1):
    """
    Run reconstruction on a few test batches, compute MSE, and save results.
    """
    model.eval()
    os.makedirs(outdir, exist_ok=True)

    total_mse = 0.0
    total_samples = 0

    for batch_idx, (x, _) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        x = x.to(device)
        x_hat, _, _ = model(x, determ=True)

        # Compute MSE for this batch
        mse = torch.mean((x_hat - x) ** 2).item()
        total_mse += mse * x.size(0)
        total_samples += x.size(0)

        # Interleave GT and recon row by row
        n = x.size(0)
        interleaved = []
        for i in range(0, n, 8):  # group into chunks of 8
            interleaved.append(x[i:i+8])
            interleaved.append(x_hat[i:i+8])
        interleaved = torch.cat(interleaved, dim=0)
        utils.save_image(interleaved, f"{outdir}/recon_batch_{batch_idx}.png", nrow=8)

        print(f"[INFO] Batch {batch_idx}: MSE = {mse:.6f}")

    # Compute overall average MSE
    avg_mse = total_mse / total_samples if total_samples > 0 else float('nan')
    print(f"[INFO] Average Reconstruction MSE (over {total_samples} samples): {avg_mse:.6f}")
    print(f"[INFO] Reconstructions saved at: {outdir}")

def load_checkpoint(model, path, device):
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"[INFO] Loaded model checkpoint from {path} (epoch {checkpoint['epoch']})")
    else:
        raise FileNotFoundError(f"[ERROR] No checkpoint found at {path}")

def main():
    parser = argparse.ArgumentParser(description="ConvVAE Testing Script")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"])
    parser.add_argument('--dataset_root', type=str, default="/scratch/chsuae/datasets")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--latent_ch", type=int, default=4)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--resume", type=str, required=True, help="Path to checkpoint to load")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to dump evaluation results")
    args = parser.parse_args()

    # Setup device and dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()

    if args.dataset == "mnist":
        dataset_dir = os.path.join(args.dataset_root, "MNIST", "test")
        test_dataset = datasets.MNIST(dataset_dir, train=False, transform=transform, download=False)
        in_ch = out_ch = 1
    else:
        dataset_dir = os.path.join(args.dataset_root, "CIFAR10", "test")
        test_dataset = datasets.CIFAR10(dataset_dir, train=False, transform=transform, download=False)
        in_ch = out_ch = 3

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Build model and load checkpoint
    model = ConvVAE(in_ch=in_ch, out_ch=out_ch, latent_ch=args.latent_ch, base_ch=args.base_ch).to(device)
    load_checkpoint(model, args.resume, device)
    
    # Evaluate reconstruction
    recon_dir = os.path.join(args.output_dir, args.dataset)
    evaluate_reconstruction(model, test_loader, device, recon_dir)

if __name__ == "__main__":
    main()
