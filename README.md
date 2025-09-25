# video_synthetic_tasks

```bash
python3 -m venv /ubc/cs/research/plai-scratch/chsu35/virtual_envs/dummy_test
source /ubc/cs/research/plai-scratch/chsu35/virtual_envs/dummy_test/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install numpy
pip install mamba-ssm[causal-conv1d] 
pip install scipy imageio matplotlib seaborn wandb
pip freeze > requirements.txt
```

# Canada Compute
Download MNIST/CIFAR10/dSprites locally and scp to Compute Canada.

