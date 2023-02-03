import yaml
import wandb
import random
import string
import json
import os

with open("sweep.yaml") as file:
    config = yaml.load(file, Loader=yaml.Loader)

sweep_id = wandb.sweep(config, project="3dunet_sweep")


def train(confi=None):
    with wandb.init(config=confi):
        name = "".join(random.choices(string.ascii_lowercase, k=5))
        keys = wandb.config.keys()
        res = {keys[i]: wandb.config[keys[i]] for i in range(len(keys))}
        with open(os.path.join("configs", name + ".json"), "w") as fp:
            json.dump(res, fp)


wandb.agent(sweep_id, train, count=1)
