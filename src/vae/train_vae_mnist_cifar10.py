import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import argparse, os
import wandb
import functools

# -------------------------
# Building Blocks
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )
    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    def __init__(self, in_ch=3, latent_ch=4, base_ch=32):
        super().__init__()
        self.down1 = nn.Sequential(
            ConvBlock(in_ch, base_ch),
            ConvBlock(base_ch, base_ch),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1),
            ConvBlock(base_ch * 2, base_ch * 2),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1),
            ConvBlock(base_ch * 4, base_ch * 4),
        )
        self.mu = nn.Conv2d(base_ch * 4, latent_ch, 1)
        self.logvar = nn.Conv2d(base_ch * 4, latent_ch, 1)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return self.mu(x), self.logvar(x)

class Decoder(nn.Module):
    def __init__(self, out_ch=3, latent_ch=4, base_ch=32):
        super().__init__()
        self.up1 = nn.Sequential(
            ConvBlock(latent_ch, base_ch * 4),
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1),
            nn.SiLU()
        )
        self.up2 = nn.Sequential(
            ConvBlock(base_ch * 2, base_ch * 2),
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1),
            nn.SiLU()
        )
        self.final = nn.Sequential(
            ConvBlock(base_ch, base_ch),
            nn.Conv2d(base_ch, out_ch, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.up1(z)
        z = self.up2(z)
        return self.final(z)

class ConvVAE(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, latent_ch=4, base_ch=32):
        super().__init__()
        self.encoder = Encoder(in_ch, latent_ch, base_ch)
        self.decoder = Decoder(out_ch, latent_ch, base_ch)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def loss_function(x_hat, x, mu, logvar, kld_coef):
    recon_loss = F.mse_loss(x_hat, x, reduction="mean")
    kld = 0.5 * torch.mean(mu.pow(2) + logvar.exp() - 1 - logvar)
    return recon_loss + kld_coef * kld, recon_loss, kld

# -------------------------
# Training & Testing
# -------------------------
def train_epoch(model, dataloader, optimizer, loss_func, device):
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kld = 0
    for x, _ in dataloader:
        x = x.to(device)
        optimizer.zero_grad()
        x_hat, mu, logvar = model(x)
        loss, recon_loss, kld = loss_func(x_hat, x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_recon_loss += recon_loss.item() * x.size(0)
        total_kld += kld.item() * x.size(0)
    return total_loss / len(dataloader.dataset), total_recon_loss, total_kld

def test_epoch(model, dataloader, loss_func, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            loss, _, _ = loss_func(x_hat, x, mu, logvar)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)

def save_reconstructions(model, dataloader, device, epoch, outdir):
    model.eval()
    os.makedirs(outdir, exist_ok=True)
    x, _ = next(iter(dataloader))
    x = x.to(device)
    with torch.no_grad():
        x_hat, _, _ = model(x)
    for i in range(4): # minibatches in 2x8 fashion
        grid = torch.cat([x[8*i:8*(i+1)], x_hat[8*i:8*(i+1)]], dim=0)
        path = f"{outdir}/recon_epoch_{epoch}_minibatch_{i}.png"
        utils.save_image(grid, path, nrow=8)
    print(f"Reconstructions saved at {outdir}.")

def save_samples(model, device, epoch, latent_shape, outdir):
    model.eval()
    os.makedirs(outdir, exist_ok=True)
    with torch.no_grad():
        z = torch.randn(16, *latent_shape, device=device)
        samples = model.decoder(z)
    path = f"{outdir}/samples_epoch_{epoch}.png"
    utils.save_image(samples, path, nrow=8)
    print(f"Samples saved at {outdir}.")

def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved at {path}.")

def load_checkpoint(model, optimizer, path, device):
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        print(f"Resumed from checkpoint {path} (epoch {checkpoint['epoch']})")
        return checkpoint["epoch"]
    else:
        print(f"No checkpoint found at {path}, starting from scratch.")
        return 0

# -------------------------
# Main with argparse
# -------------------------
def main():
    import wandb, socket, json

    def setup_wandb(config: dict, wandb_config: dict, run_name: str = None, mode: str = None):
        """
        Initializes wandb with automatic offline fallback if internet is unavailable.
        """
        # Auto-detect if we have network access
        if mode is None:
            try:
                socket.create_connection(("api.wandb.ai", 443), timeout=3)
                mode = "online"
            except OSError:
                print("[INFO] No internet detected: running wandb in offline mode.")
                mode = "offline"

        os.environ["WANDB_MODE"] = mode  # makes wandb respect mode globally
        wandb.login(key=wandb_config["api_key"])
        wandb.init(
            project=wandb_config["project"],
            entity=wandb_config["entity"],  
            name=run_name,
            mode=mode,  # redundant but explicit
            config=config
        )
        print(f"[INFO] wandb initialized in {mode} mode. Run URL: {wandb.run.get_url() if wandb.run else 'N/A'}.")
    
    parser = argparse.ArgumentParser(description="ConvVAE for MNIST/CIFAR10")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"])
    parser.add_argument('--dataset_root', type=str, default="/scratch/chsuae/datasets")
    parser.add_argument('--wandb_config', type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--latent_ch", type=int, default=4)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kld_coef", type=float, default=1e-2)
    parser.add_argument("--output_dir", type=str, default="/scratch/chsuae/")
    parser.add_argument("--save_interval", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--dump_interval", type=int, default=5, help="Dump recon/samples every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    run_name = f"vae-{args.dataset}-lr{args.lr}-b{args.batch_size}-kld{args.kld_coef}"
    output_dir = os.path.join(args.output_dir, run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()

    if args.dataset == "mnist":
        dataset_dir = os.path.join(args.dataset_root, "MNIST")
        os.makedirs(dataset_dir, exist_ok=True)
        train_dataset = datasets.MNIST(os.path.join(dataset_dir, "train"), train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(os.path.join(dataset_dir, "test"), train=False, transform=transform, download=True)
        in_ch = out_ch = 1
    else:
        dataset_dir = os.path.join(args.dataset_root, "CIFAR10")
        os.makedirs(dataset_dir, exist_ok=True)
        train_dataset = datasets.CIFAR10(os.path.join(dataset_dir, "train"), train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(os.path.join(dataset_dir, "test"), train=False, transform=transform, download=True)
        in_ch = out_ch = 3

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = ConvVAE(in_ch=in_ch, out_ch=out_ch, latent_ch=args.latent_ch, base_ch=args.base_ch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    dummy_input = torch.zeros(1, in_ch, *train_dataset[0][0].shape[1:], device=device)
    with torch.no_grad():
        mu, _ = model.encoder(dummy_input)
    latent_shape = mu.shape[1:]

    print(f"Loading wandb config file: {args.wandb_config}")
    with open(args.wandb_config) as f:
        wandb_config = json.load(f)
    setup_wandb(config=vars(args), 
                wandb_config=wandb_config,
                run_name=run_name,
                mode="offline"
                )
    print(f"VAE Config: {vars(args)}")

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume, device)
    
    loss_func = functools.partial(loss_function, kld_coef=args.kld_coef)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_loss, train_recon_loss, train_kld = train_epoch(model, train_loader, optimizer, loss_func, device)
        test_loss = test_epoch(model, test_loader, loss_func, device)
        wandb.log({"train_loss": train_loss, 
                   "train_recon_loss": train_recon_loss, 
                   "train_kld": train_kld, 
                   "test_loss": test_loss
                   }, step=epoch
                   )
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, \
                Train Recon Loss = {train_recon_loss:.4f}, \
                Train KLD = {train_kld:.4f}, \
                Test Loss = {test_loss:.4f}."
                )

        if epoch % args.dump_interval == 0:
            print(f"Epoch {epoch}, dumping results...")
            save_reconstructions(model, test_loader, device, epoch, os.path.join(output_dir, "recon"))
            save_samples(model, device, epoch, latent_shape, os.path.join(output_dir, "samples"))

        if epoch % args.save_interval == 0:
            print(f"Epoch {epoch}, saving checkpoint...")
            ckpt_path = os.path.join(output_dir, "checkpoints", f"vae_epoch_{epoch}.pt")
            save_checkpoint(model, optimizer, epoch, ckpt_path)

if __name__ == "__main__":
    main()
