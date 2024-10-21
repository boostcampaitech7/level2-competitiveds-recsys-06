import datetime
import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import wandb

from src.model.simplemodel import SimpleModel

from src.custom_wandb import WandB

from src.config import get_config


def app():
    _ = WandB()
    config = get_config()
    server_conf = config.get("server")
    run_name = f"{server_conf.get('number')}-{server_conf.get('model_type')}-{server_conf.get('mode')}"
    with wandb.init() as run:
        print(f"start {run_name}")
        print(f"config : {config}")
        run.log({"config": config})
        run.name = run_name
        run._notes = config
        model = SimpleModel(config)
        model.train()
        pred = model.predict()
        filepath = os.path.join(
            config.get("data").get("output_path"),
            f"{run_name}-{datetime.datetime.now()}.csv",
        )
        pred.to_csv(filepath, index=False)


if __name__ == "__main__":
    app()
