from typing import List

import numpy as np
import pandas as pd
import wandb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from xgboost.callback import TrainingCallback

from src.config import get_config
from src.model import print_evaluation
from src.model.interface import ModelInterface
from src.model.valid.KFold import CustomKFold


class Model(ModelInterface):
    """
    XGBoost model

    xgb에선 xgb.DMatrix를 사용합니다. (lgbm에선 lgb.Dataset을 사용)
    xbgoost 사용 시 기본적으로 수치형데이터만 입력으로 받으므로 범주형 변수를 미리 인코딩 해야합니다.
    결측치는 자동으로 처리하므로 결측치가 많지 않은 데이터는 따로 처리하지 않아도 됩니다.
    feature importance를 확인할 수 있고, early stopping을 사용할 수 있습니다.

    """

    def __init__(self, x_train: pd.DataFrame, y_train: pd.DataFrame, config: any):
        super().__init__(x_train, y_train, config)
        self.model: List[xgb.Booster] = []

    def _convert_pred_dataset(self, df):
        return xgb.DMatrix(df)

    def get_model(self) -> List[xgb.Booster]:
        return self.model

    def train(self):
        try:
            # XGBoost를 위한 DMatrix 생성
            d_train = xgb.DMatrix(self.x_train, label=self.y_train)
            model = xgb.train(
                self.hyper_params,
                d_train,
                num_boost_round=self.hyper_params.get("num_boost_round"),
                early_stopping_rounds=self.hyper_params.get("early_stopping_round"),
                verbose_eval=self.hyper_params.get("verbose_eval"),
                callbacks=[WandbCallback()],
            )
            self.model.append(model)

        except Exception as e:
            print(e)

    def train_with_kfold(self) -> None:
        try:
            kf = CustomKFold().get_fold()
            print(f"Feature Column is {self.x_train.columns}")

            # 각 폴드의 예측 결과를 저장할 리스트
            oof_predictions = np.zeros(len(self.x_train))
            num_boost_round = self.hyper_params.pop("num_boost_round")
            early_stopping_rounds = self.hyper_params.pop("early_stopping_rounds")
            verbose_eval = self.hyper_params.pop("verbose_eval")
            wandb_callback = WandbCallback()
            # 교차 검증 수행
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.x_train), 1):
                print(f"Fold-{fold} is Start")
                x_train, x_val = (
                    self.x_train.iloc[train_idx],
                    self.x_train.iloc[val_idx],
                )
                y_train, y_val = (
                    self.y_train.iloc[train_idx],
                    self.y_train.iloc[val_idx],
                )
                # XGBoost를 위한 DMatrix 생성
                d_train = xgb.DMatrix(x_train, label=y_train)
                d_val = xgb.DMatrix(x_val, label=y_val)
                # XGBoost 파라미터 설정
                # 모델 학습
                evals = [(d_train, "train"), (d_val, "eval")]
                model = xgb.train(
                    self.hyper_params,
                    d_train,
                    num_boost_round=num_boost_round,
                    early_stopping_rounds=early_stopping_rounds,
                    evals=evals,
                    verbose_eval=verbose_eval,
                    callbacks=[wandb_callback],
                )
                self.model.append(model)
                # 검증 세트에 대한 예측
                oof_predictions[val_idx] = model.predict(d_val)
            oof_mae = mean_absolute_error(self.y_train, oof_predictions)
            print(f"MAE:{oof_mae:.4f}")
            wandb.log({"MAE": f"{oof_mae:.4f}"})

            oof_mae = mean_absolute_error(self.y_train, np.expm1(oof_predictions))
            print(f"Log-T-MAE:{oof_mae:.4f}")
            wandb.log({"Log-T-MAE": f"{oof_mae:.4f}"})
        except Exception as e:
            print(e)


# Custom W&B Callback 정의
class WandbCallback(TrainingCallback):
    def __init__(self):
        super().__init__()
        self.log_metrics = [get_config().get("xgboost").get("eval-metric")]
        self.log_interval = get_config().get("print").get("evaluation-period")
        self.fold_subfix = ""
        self.n_fold = 0
        self.current_step = 0

    def after_iteration(self, model, epoch, evals_log):
        """매 부스팅 라운드 이후 실행되는 메서드"""
        if (epoch + 1) % self.log_interval == 0:
            metrics_to_log = {}
            for dataset, metric_dict in evals_log.items():
                for metric, values in metric_dict.items():
                    if metric in self.log_metrics:
                        metrics_to_log[f"{dataset}_{metric}"] = values[-1]
            self.current_step += epoch + 1
            wandb.log(metrics_to_log, step=self.current_step)

        return False  # 학습 계속

    def after_training(self, model):
        """학습 종료 후 Feature Importance 기록"""
        # Feature Importance 추출
        importance_types = ["weight", "gain", "cover"]
        for imp_type in importance_types:
            feature_importance = model.get_score(importance_type=imp_type)
            sorted_importance = dict(
                sorted(
                    feature_importance.items(), key=lambda item: item[1], reverse=True
                )
            )

            # W&B에 Feature Importance 기록
            wandb.log(
                {f"feature_importance_{imp_type}_{self.n_fold}": sorted_importance}
            )
            self.n_fold += 1
        return model
