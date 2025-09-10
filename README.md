# video_synthetic_tasks

## Create virtual environment

```bash
python3 -m venv /ubc/cs/research/plai-scratch/chsu35/virtual_envs/synth_tasks
source /ubc/cs/research/plai-scratch/chsu35/virtual_envs/synth_tasks/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy scipy tqdm
pip install imageio
pip install matplotlib seaborn
pip install wandb

pip freeze > requirements.txt
```


