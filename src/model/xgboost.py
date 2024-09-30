import os.path
from typing import Dict

import joblib
import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from src.model.interface import ModelInterface


class Model(ModelInterface):
    """
    XGBoost model

    xgb에선 xgb.DMatrix를 사용합니다. (lgbm에선 lgb.Dataset을 사용)
    xbgoost 사용 시 기본적으로 수치형데이터만 입력으로 받으므로 범주형 변수를 미리 인코딩 해야합니다.
    결측치는 자동으로 처리하므로 결측치가 많지 않은 데이터는 따로 처리하지 않아도 됩니다.
    feature importance를 확인할 수 있고, early stopping을 사용할 수 있습니다.

    train() = 모델 트레이닝 입니다.
    train_validation() = Validation을 위한 모델 트레이닝 입니다.
    predict() = train() 혹은 train_validation() 후, 예측을 위한 메서드 입니다. - 모델이 없을 경우(오류 발생)
                / 모델이 있을 경우(결과 & print("validation" or "train") - 모드)

    """

    def __init__(self, x_train: pd.DataFrame, y_train: pd.DataFrame, config: any):
        self.x_train: pd.DataFrame = x_train
        self.y_train: pd.DataFrame = y_train
        self.model: xgb.Booster | None = None
        self.config: Dict[any] = config
        params: Dict = self.config.get("xgboost")
        params["random_state"] = self.config.get("data").get("random_state")
        self.hyper_params = params
        self.mode: str | None = None

    def train(self):
        try:
            self.mode = "train"
            x_train = self.x_train
            y_train = self.y_train
            dtrain = xgb.DMatrix(x_train, label=y_train)
            # xgb train
            self.model = xgb.train(
                params=self.hyper_params,
                dtrain=dtrain,
            )
        except Exception as e:
            print(e)
            self._reset_model()

    def predict(self, test_df: pd.DataFrame):
        if self.model is not None:
            print(f"This model is **{self.mode}**.")
            # 입력을 DMatrix 형태로
            dtest = xgb.DMatrix(test_df)
            return self.model.predict(dtest)
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

            # xgb DMatrix
            dtrain = xgb.DMatrix(x_train, label=y_train)
            dvalid = xgb.DMatrix(x_valid, label=y_valid)

            # xgb train
            self.model = xgb.train(
                params=self.hyper_params,
                dtrain=dtrain,
                evals=[(dvalid, "eval")],
            )

            # xgb predict
            y_valid_pred = self.predict(x_valid)
            y_valid_pred_class = np.argmax(
                y_valid_pred, axis=1
            )  # multi:softprob 모드 사용시 class로 변환 필요

            # score check
            accuracy = accuracy_score(y_valid, y_valid_pred_class)
            auroc = roc_auc_score(y_valid, y_valid_pred, multi_class="ovr")
            f1 = f1_score(y_valid, y_valid_pred_class, average="weighted")

            print(f"acc: {accuracy}, auroc: {auroc}, f1: {f1}")
        except Exception as e:
            print(e)
            self._reset_model()

    def _reset_model(self):
        self.mode = None
        self.model = None

    def export_model(self, dir_path):
        joblib.dump(self.model, os.path.join(dir_path, "xgboost.pkl"))
