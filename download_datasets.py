import os
import urllib.request
import torchvision
import torchvision.transforms as transforms

# ===========================
# Base directory for all datasets
# ===========================

# UBC cluster, not possible to download from internet on Compute Canada clusters
# base_dir = "/ubc/cs/research/plai-scratch/chsu35/datasets"

base_dir = ...
os.makedirs(base_dir, exist_ok=True)

# Dataset-specific directories
mnist_dir = os.path.join(base_dir, "MNIST")
cifar_dir = os.path.join(base_dir, "CIFAR10")
dsprites_dir = os.path.join(base_dir, "dSprites")

os.makedirs(mnist_dir, exist_ok=True)
os.makedirs(cifar_dir, exist_ok=True)
os.makedirs(dsprites_dir, exist_ok=True)

# ===========================
# Common transforms
# ===========================
transform = transforms.Compose([
    transforms.ToTensor(),   # Converts to [0,1] float tensor
])

# ===========================
# MNIST
# ===========================
print("Downloading MNIST...")
torchvision.datasets.MNIST(root=os.path.join(mnist_dir, "train"), train=True, download=True, transform=transform)
torchvision.datasets.MNIST(root=os.path.join(mnist_dir, "test"), train=False, download=True, transform=transform)
print("MNIST downloaded.")

# ===========================
# CIFAR-10
# ===========================
print("Downloading CIFAR-10...")
torchvision.datasets.CIFAR10(root=os.path.join(cifar_dir, "train"), train=True, download=True, transform=transform)
torchvision.datasets.CIFAR10(root=os.path.join(cifar_dir, "test"), train=False, download=True, transform=transform)
print("CIFAR-10 downloaded.")

# ===========================
# dSprites
# ===========================
dsprites_url = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
dsprites_file = os.path.join(dsprites_dir, "dsprites.npz")

if not os.path.exists(dsprites_file):
    print("Downloading dSprites...")
    urllib.request.urlretrieve(dsprites_url, dsprites_file)
    print("dSprites downloaded.")
else:
    print("dSprites already exists.")
