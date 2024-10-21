import wandb

from src.config import get_config


def print_evaluation():
    period = get_config().get("print").get("evaluation-period")

    def callback(env):
        if (env.iteration + 1) % period == 0:
            train_mae = env.evaluation_result_list[0][2]
            val_mae = env.evaluation_result_list[1][2]
            wandb.log({"train_mae": train_mae, "val_mae": val_mae})
            print(
                f"[{env.iteration + 1}] Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}"
            )

    return callback
