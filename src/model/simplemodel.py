from typing import Type, Tuple, List

import pandas as pd
from numpy import ndarray

from src.pre_process.interface import PreProcessInterface
from src.model.interface import ModelInterface


class SimpleModel:
    def __init__(
        self,
        data: pd.DataFrame = None,
        model_type: str = None,
        pre_process_type: List[str] = None,
    ):
        self._get_config()  # 하이퍼 파라미터 부분 및 기타 설정 - config-sample.yaml 수정에 따라 사용
        if data is None:
            data = pd.read_csv(self.config.get("data").get("path"))
        self.data = data
        if model_type is None:
            self.model_type = self.config.get("server").get("model_type")
        if pre_process_type is None:
            self.pre_process_types: List[str] = self.config.get("server").get(
                "pre_process_type"
            )
        self.model: ModelInterface
        self.train_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None
        # 데이터 전처리
        self.pre_process_list: List[Type[PreProcessInterface]] = self._get_pre_process()
        # 학습을 위한 부분
        self._set_model_preprocess()

    def _get_config(self):
        from src.config import get_config

        self.config = get_config()

    def _set_model_preprocess(self):
        if self.train_df is None or self.test_df is None:
            self._update_data_by_pp()
        model = self._get_model()(
            self.train_df.drop(["target", "ID"], axis=1, errors="ignore"),
            self.train_df["target"],
            self.config,
        )
        self.model = model
        return

    def _update_data_by_pp(self):
        pre_precess_list = self._get_pre_process()
        for pre_precess in pre_precess_list:
            pp = pre_precess(self.data)
            self.data = pp.get_data()
        self.train_df, self.test_df = pp.get_train_test()
        return

    def _get_model(self) -> Type[ModelInterface]:
        # Model Class Import
        from src.model.lgbm import Model as lgbm
        from src.model.xgboost import Model as xgb

        # 타입별 모델 변경
        model_type = self.model_type.lower()
        if model_type == "baseline":
            return lgbm
        elif model_type == "xgboost":
            return xgb
        else:
            raise Exception(f"{self.model_type}: 해당 모델은 지원되지 않습니다.")

    def _get_pre_process(self) -> list[Type[PreProcessInterface]]:

        # Type별 Class 전처리 추가
        pp_list = []
        for pre_process_type in self.pre_process_types:
            if pre_process_type.lower() == "baseline":
                pp_list.append(BaseLineData)

            elif pre_process_type.lower() == "final":
                pp_list.append(final_pp)
            else:
                raise Exception(f"{self.model_type}: 해당 모델은 지원되지 않습니다.")
        return pp_list

    def train(self):
        self.model.train()

    def train_for_validation(self):
        self.model.train_validation()

    def predict(self) -> ndarray | list:
        return self.model.predict(
            self.test_df.drop(["target", "ID"], axis=1, errors="ignore")
        )

    def get_train_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.train_df, self.test_df

    def get_model(self, dir_path: str) -> ModelInterface:
        return self.model.export_model(dir_path)
