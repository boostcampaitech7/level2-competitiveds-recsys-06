import os.path
from typing import Dict

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import Booster
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold

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

    def get_model(self):
        assert self.model is not None
        return self.model

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
            feature_columns = data_config["features"]

            # train_test_split 으로 valid set, train set 분리

            # 콜백 함수 정의
            def print_evaluation(period=10):
                def callback(env):
                    if (env.iteration + 1) % period == 0:
                        train_mae = env.evaluation_result_list[0][2]
                        val_mae = env.evaluation_result_list[1][2]
                        print(
                            f"[{env.iteration + 1}] Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}"
                        )

                return callback

            # 5-fold 교차 검증 설정
            n_folds = 5
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

            # 각 폴드의 예측 결과를 저장할 리스트
            oof_predictions = np.zeros(len(self.x_train))

            # 교차 검증 수행
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.x_train), 1):
                print(f"\nFold {fold}")

                X_train, X_val = (
                    self.x_train.iloc[train_idx],
                    self.x_train.iloc[val_idx],
                )
                y_train, y_val = (
                    self.y_train.iloc[train_idx],
                    self.y_train.iloc[val_idx],
                )

                dtrain = lgb.Dataset(X_train, label=y_train)
                dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

                model = lgb.train(
                    self.hyper_params,
                    dtrain,
                    num_boost_round=1000,
                    valid_sets=[dtrain, dval],
                    callbacks=[print_evaluation(period=100)],
                )

                # 검증 세트에 대한 예측
                oof_predictions[val_idx] = model.predict(X_val)

                # 테스트 세트에 대한 예측 - 현행 인터페이스에 맞게 별도 분리 필요
                # test_predictions += model.predict(test_df[feature_columns]) / n_folds

            # 전체 검증 세트에 대한 MAE 계산
            oof_mae = mean_absolute_error(self.y_train, oof_predictions)
            print(f"\nOverall OOF MAE: {oof_mae:.4f}")

        except Exception as e:
            print(e)
            self._reset_model()

    def _reset_model(self):
        self.mode = None
        self.model = None

    def export_model(self, dir_path):
        joblib.dump(self.model, os.path.join(dir_path, "lightgbm.pkl"))
