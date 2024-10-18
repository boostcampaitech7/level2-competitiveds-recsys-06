import pandas as pd
from sklearn.model_selection import KFold
import numpy as np

from src.config import get_config

import wandb


class CustomKFold(object):
    def __init__(self):
        config = get_config()
        data_conf = config.get("data")
        fold_conf = data_conf.get("k-fold")
        self.n_folds = fold_conf.get("n_folds")
        self.shuffle = fold_conf.get("shuffle")
        self.random_state = data_conf.get("random_state")

    def get_fold(self) -> KFold:
        kf = KFold(
            n_splits=self.n_folds,
            shuffle=bool(self.shuffle),
            random_state=self.random_state,
        )
        return kf
