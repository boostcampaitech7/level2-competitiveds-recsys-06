import pandas as pd
import numpy as np

from typing import List
from src.pre_process.interface import PreProcessInterface
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


class ArimaDepositIndexFeature(PreProcessInterface):
    """
    df: _type에 train, test가 포함된 데이터프레임
    interestRate: raw의 interestRate 데이터프레임

    deposit의 시계열 데이터를 기반으로 ARIMA 모델을 학습하여
    deposit의 예측 지수를 생성
    (이자율을 반영하는 sarimax 지수는 추후 eda 검토 후 추가 할수도 있음)
    """

    def __init__(self, df: pd.DataFrame, interestRate: pd.DataFrame, **kwargs):
        self.interest_rate_df = interestRate
        super().__init__(df, **kwargs)

    def get_train_test(self) -> (pd.DataFrame, pd.DataFrame):
        pass

    def get_data(self) -> pd.DataFrame:
        return self.df

    def _preprocess(self):
        self.make_month_df()
        self.make_arima_deposit_index()
        pass

    def make_arima_deposit_index(self):
        month_df = self.month_df
        df = self.df

        # NaN 값을 가진 행을 제외하고 ARIMA 모델을 훈련
        train_df = month_df.dropna(subset=["avg_deposit"])

        # ARIMA 모델을 사용하여 avg_deposit 예측
        model = ARIMA(train_df["avg_deposit"], order=(6, 1, 1))
        model_fit = model.fit()

        # NaN 값을 가진 행의 인덱스를 찾기
        nan_indices = month_df[month_df["avg_deposit"].isna()].index

        # NaN 값을 가진 행의 avg_deposit 예측
        predictions = model_fit.predict(
            start=len(train_df), end=len(train_df) + len(nan_indices) - 1, typ="levels"
        )

        # 예측된 값을 month_df에 채우기
        month_df.loc[nan_indices, "avg_deposit"] = predictions.values

        # 가장 이른 시점의 avg_deposit 값을 기준으로 설정
        base_value = month_df["avg_deposit"].iloc[0]

        # arima_deposit_index 컬럼 생성
        month_df["arima_deposit_index"] = (month_df["avg_deposit"] / base_value) * 100

        # df의 contract_year_month를 datetime 형식으로 변환
        df["contract_ymd"] = pd.to_datetime(
            df["contract_year_month"], format="%Y%m", errors="coerce"
        )

        # month_df의 contract_ymd와 df의 contract_ymd를 기준으로 병합하여 arima_deposit_index를 df에 추가
        df = pd.merge(
            df,
            month_df[["contract_ymd", "arima_deposit_index"]],
            on="contract_ymd",
            how="left",
        )

        # 결과를 저장
        self.df = df.drop(columns=["contract_ymd"])

    def make_month_df(self):
        df = self.df
        interest_rate_df = self.interest_rate_df
        df["contract_ymd"] = pd.to_datetime(
            df["contract_year_month"], format="%Y%m", errors="coerce"
        )
        month_df = (
            df.groupby(df["contract_ymd"].dt.to_period("M"))
            .agg(avg_deposit=("deposit", "mean"))
            .reset_index()
        )

        month_df["contract_ymd"] = month_df["contract_ymd"].dt.to_timestamp()

        # 'year_month'를 datetime 형식으로 변환
        interest_rate_df["year_month"] = pd.to_datetime(
            interest_rate_df["year_month"], format="%Y%m"
        )

        # 'contract_ymd'와 'year_month'를 기준으로 병합
        month_df = pd.merge_asof(
            month_df.sort_values("contract_ymd"),
            interest_rate_df.sort_values("year_month"),
            left_on="contract_ymd",
            right_on="year_month",
            direction="backward",
        )

        # 필요 없는 'year_month' 열 삭제
        month_df.drop(columns=["year_month"], inplace=True)

        # 결측값이 있는 경우 이전 값으로 채우기
        month_df["interest_rate"] = month_df["interest_rate"].ffill()
        month_df["avg_deposit"] = month_df["avg_deposit"].replace(-999, np.nan)

        self.month_df = month_df
        self.df = df
