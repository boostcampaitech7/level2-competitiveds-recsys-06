import sys
import os

import wandb

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.config import get_config


class WandB:
    def __init__(self):
        config = get_config()
        server_name = f'server-{config.get("server").get("no")}'
        print("---- START ----")
        print(server_name)
        self.server_name = server_name
        self.w = wandb
        self.w.login(key=config.get("wandb").get("api-key"))
        print("W&B Login is completed.")
