import os.path
from typing import Dict

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import Booster
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.model.interface import ModelInterface


class Model(ModelInterface):
    """
    LightGBM model
    해당 모델은 BaseLine 모델입니다.
    BaseLine으로 제공된 코드 기반으로 작성되었습니다.

    train() = 모델 트레이닝 입니다.
    train_validation() = Validation을 위한 모델 트레이닝 입니다.
    predict() = train() 혹은 train_validation() 후, 예측을 위한 메서드 입니다. - 모델이 없을 경우(오류 발생)
                / 모델이 있을 경우(결과 & print("validation" or "train") - 모드)
    """

    def __init__(self, x_train: pd.DataFrame, y_train: pd.DataFrame, config: any):
        self.x_train: pd.DataFrame = x_train
        self.y_train: pd.DataFrame = y_train
        self.model: Booster | None = None
        self.config: Dict[any] = config
        params: Dict = self.config.get("light-gbm")
        params["random_state"] = self.config.get("data").get("random_state")
        self.hyper_params = params
        self.mode: str | None = None

    def train(self):
        try:
            self.mode = "train"
            x_train = self.x_train
            y_train = self.y_train
            train_data = lgb.Dataset(x_train, label=y_train)
            # lgb train
            self.model = lgb.train(
                params=self.hyper_params,
                train_set=train_data,
            )
        except Exception as e:
            print(e)
            self._reset_model()

    def predict(self, test_df: pd.DataFrame):
        if self.model is not None:
            print(f"This model is **{self.mode}**.")
            return self.model.predict(test_df)
        raise Exception("Model is not trained")

    def train_validation(self) -> None:
        try:
            self.mode = "valid"
            data_config = self.config.get("data")
            # train_test_split 으로 valid set, train set 분리
            x_train, x_valid, y_train, y_valid = train_test_split(
                self.x_train,
                self.y_train,
                test_size=data_config.get("valid_size"),
                random_state=data_config.get("random_state"),
            )

            # lgb dataset
            train_data = lgb.Dataset(x_train, label=y_train)
            valid_data = lgb.Dataset(x_valid, label=y_valid, reference=train_data)

            # lgb train
            self.model = lgb.train(
                params=self.hyper_params,
                train_set=train_data,
                valid_sets=valid_data,
            )

            # lgb predict
            y_valid_pred = self.predict(x_valid)
            y_valid_pred_class = np.argmax(y_valid_pred, axis=1)

            # score check
            accuracy = accuracy_score(y_valid, y_valid_pred_class)
            auroc = roc_auc_score(y_valid, y_valid_pred, multi_class="ovr")

            print(f"acc: {accuracy}, auroc: {auroc}")
        except Exception as e:
            print(e)
            self._reset_model()

    def _reset_model(self):
        self.mode = None
        self.model = None

    def export_model(self, dir_path):
        joblib.dump(self.model, os.path.join(dir_path, "lightgbm.pkl"))
