import random
import wandb


# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="plaicraft-academic",
    # Set the wandb project where this run will be logged.
    project="rnn-synthesized-task-project",
    name="dummy",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
)

# Simulate training.
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    # Log metrics to wandb.
    run.log({"acc": acc, "loss": loss})

# Finish the run and upload any remaining data.
run.finish()

'''
singularity shell --nv --bind ${HOME} "/ubc/cs/research/plai-scratch/chsu35/singularity_setup/video_synth_tasks/ubc_env.sif"
wandb sync ./wandb/offline-*
'''