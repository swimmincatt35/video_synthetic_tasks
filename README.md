# video_synthetic_tasks

## Create virtual environment

- The ```mamba-ssm``` package requires a GPU in the background in order to be installed successfully.
- The ```mamba-ssm``` package requires using older versions of torch and cuda (see https://github.com/state-spaces/mamba/issues/217). 
- Line 20, ```import selective_scan``` in ```/venv/lib/python3.12/site-packages/mamba_ssm/ops/selective_scan_interface.py``` should be commented out in the end of installation (see https://blog.csdn.net/weixin_52153243/article/details/142737403).

```bash
python3 -m venv /ubc/cs/research/plai-scratch/chsu35/virtual_envs/vid_synth_tasks
source /ubc/cs/research/plai-scratch/chsu35/virtual_envs/vid_synth_tasks/bin/activate
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install numpy scipy tqdm
pip install imageio
pip install matplotlib seaborn
pip install wandb
pip install mamba-ssm[causal-conv1d]
pip freeze > requirements.txt
```

