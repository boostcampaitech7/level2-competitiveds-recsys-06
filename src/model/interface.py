from abc import abstractmethod, ABC
from typing import Dict

import numpy as np
import pandas as pd
import os

import joblib
from sklearn.metrics import mean_absolute_error


class ModelInterface(ABC):
    """
    SimpleModel에서 사용하기 편하게 하기위한 Interface입니다.
    만드는 모델은 각각 추상화를 통해 진행 해야 합니다.
    현재 인터페이스는 반드시 상속 후 구현.

    train() = 모델 트레이닝 입니다.
    train_with_kfold() = K-Fold로 Validation 진행 및 K개의 모델을 Bagging 하는 메서드입니다.
    predict() = train() 혹은 train_with_kfold() 후, 예측을 위한 메서드 입니다. - 모델이 없을 경우(오류 발생)"""

    @abstractmethod
    def __init__(self, x_train: pd.DataFrame, y_train: pd.DataFrame, config: any):
        self.x_train: pd.DataFrame = x_train
        self.y_train: pd.DataFrame = y_train
        self.config: Dict[any] = config
        self.model_name = config.get("server").get("model_type")
        params: Dict[str, any] = self.config.get(self.model_name)
        params.update({"random_state": self.config.get("data").get("random_state")})
        self.hyper_params = params
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def train_with_kfold(self) -> None:
        pass

    @abstractmethod
    def _convert_pred_dataset(self, df):
        pass

    def predict(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        df = df.drop(columns=[target], errors="ignore")
        c_df = self._convert_pred_dataset(df)
        oof_pred = np.zeros(df.shape[0])
        model = self.get_model()
        for e in model:
            oof_pred += e.predict(c_df)
        pred = oof_pred / len(model)
        df["pred"] = pred
        df.reset_index(inplace=True, drop=False)
        return df[["index", "pred"]]

    @abstractmethod
    def get_model(self):
        pass

    def export_model(self, dir_path):
        for i, e in enumerate(self.get_model()):
            joblib.dump(e, os.path.join(dir_path, f"{self.model_name}-K-{i}.pkl"))
