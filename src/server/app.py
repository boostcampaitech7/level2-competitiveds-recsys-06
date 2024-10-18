import datetime
from typing import Dict
import sys
import os

import wandb

from src.model.simplemodel import SimpleModel

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.custom_wandb import WandB

from src.config import get_config


def app():
    _ = WandB()
    config = get_config()
    server_conf = config.get("server")
    run_name = f"{server_conf.get("no")}-{server_conf.get("model_type")}-{server_conf.get('mode')}"
    with wandb.init() as run:
        run.name = run_name
        run.notes = config
        model = SimpleModel(config)
        model.train()


if __name__ == "__main__":
    app()
