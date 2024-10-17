import datetime
from builtins import function
from typing import Dict
import sys
import os

import wandb

from src.model.interface import ModelInterface

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
        self.w.login(key=config.get("wandb").get("api_key"))
        self.config = None

    def set_config(self, config: Dict[any, any]):
        self.config = config

    def log(self):
        pass

    def model_with_wandb(self, model_class: [ModelInterface], is_valid: bool):
        # config 값 있는지 처리
        assert self.config is not None
        model_fit_type = "Test"
        if is_valid:
            model_fit_type = "Valid"
        with self.w.init(project=self.server_name) as run:
            model = model_class()
            prj_name = model.get_name()  # Model Name
            run.name = prj_name
            run.config.update(self.config)
            run.notes = f"""
            ```json
            {self.config}
            ```
            """
            # Model 적용
            model = model_class()
            run.watch(model)
            log = {
                f"{model_fit_type} Accuracy": acc,
                f"{model_fit_type} Loss": loss,
            }
            if model_fit_type == "Valid":
                log["Valid Method"] = "K-Fold"
            wandb.log(log)
