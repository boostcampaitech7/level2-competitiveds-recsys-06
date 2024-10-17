import datetime
from typing import Dict
import sys
import os

import wandb

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.config import get_config


def app():
    config = get_config()
    server_name = f'server-{config.get("server").get("no")}'
    print("---- START ----")
    print(server_name)
    wandb.login(key=config.get("wandb").get("api_key"))
    # 프로젝트 반복문
    #  model_test_code(wandb, server_name, {"": ""})  # config

    return


def model_test_code(w, server_name: str, config: Dict[any, any]):
    model_config = config  # config.get("~")
    ## Model 시작전
    with wandb.init(project=server_name) as run:
        prj_name = ""  # Model Name
        run.name = prj_name
        run.config.update(model_config)
        run.notes = f"""
        ```json
        {model_config}
        ```
        """
        # Model 적용
        model = None
        run.watch(model)
        # wandb.log({
        #     #"Examples": example_images, #
        #     "Valid Method: config.get("server").get("valid_type")
        #     "Valid Accuracy": valid_acc,
        #     "Valid Loss": valid_loss})

        # wandb.log({
        #     #"Examples": example_images, #
        #     "Test Accuracy": test_acc,
        #     "Test Loss": test_loss})


if __name__ == "__main__":
    app()
