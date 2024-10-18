from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd
import wandb
from lightgbm import Booster
from sklearn.metrics import mean_absolute_error

from src.model import print_evaluation
from src.model.interface import ModelInterface
from src.model.valid.KFold import CustomKFold


class Model(ModelInterface):
    """
    LightGBM model
    해당 모델은 BaseLine 모델입니다.
    """

    def _convert_pred_dataset(self, df):
        pass

    def __init__(self, x_train: pd.DataFrame, y_train: pd.DataFrame, config: any):
        super().__init__(x_train, y_train, config)
        self.model: List[Booster] | Booster | None = None

    def train(self):
        try:
            train_data = lgb.Dataset(self.x_train, label=self.y_train)
            # lgb train
            self.model = lgb.train(
                params=self.hyper_params,
                train_set=train_data,
                num_boost_round=self.hyper_params.get("num_boost_round"),
                callbacks=[print_evaluation()],
            )
        except Exception as e:
            print(e)
            self._reset_model()

    def train_with_kfold(self) -> None:
        try:
            kf = CustomKFold().get_fold()

            # 각 폴드의 예측 결과를 저장할 리스트
            oof_predictions = np.zeros(len(self.x_train))

            # 교차 검증 수행
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.x_train), 1):
                if self.model is None or self.model:
                    self.model = []
                x_train, x_val = (
                    self.x_train.iloc[train_idx],
                    self.x_train.iloc[val_idx],
                )
                y_train, y_val = (
                    self.y_train.iloc[train_idx],
                    self.y_train.iloc[val_idx],
                )

                d_train = lgb.Dataset(x_train, label=y_train)
                d_val = lgb.Dataset(x_val, label=y_val, reference=d_train)

                model = lgb.train(
                    self.hyper_params,
                    d_train,
                    num_boost_round=self.hyper_params.get("num_boost_round"),
                    valid_sets=[d_train, d_val],
                    callbacks=[print_evaluation()],
                )
                self.model.append(model)
                # 검증 세트에 대한 예측
                oof_predictions[val_idx] = model.predict(x_val)

            # 전체 검증 세트에 대한 MAE 계산
            oof_mae = mean_absolute_error(self.y_train, oof_predictions)
            wandb.log({"MAE": f"{oof_mae:.4f}"})

        except Exception as e:
            print(e)
            self._reset_model()
