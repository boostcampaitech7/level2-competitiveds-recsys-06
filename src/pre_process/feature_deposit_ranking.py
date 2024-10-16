import pandas as pd

from typing import List
from src.pre_process.interface import PreProcessInterface


class FeatureAptDepositRanking(PreProcessInterface):
    """
    아파트별 평균 전세가 랭킹을 생성하는 클래스.
    _type이 붙은 채로 train과 test를 concat한 뒤에 이 클래스에 넣어야 한다.

    apt_deposit_rank : 위도, 경도가 같은 raw끼리 묶어 그룹화한 뒤 평균 전세가격순으로 랭킹을 매긴 피처.
    test에서 평균이 nan인 그룹은 평균 deposit으로 대체해서 랭킹 생성

    apt_area_deposit_rank :  위도, 경도+면적까지 같은 raw끼리 묶어 그룹화한 뒤 평균 전세가격순으로 랭킹을 매긴 피처.
    test에서 평균이 nan인 그룹은 평균 deposit으로 대체해서 랭킹 생성
    """

    def __init__(self, df: pd.DataFrame):
        super(FeatureAptDepositRanking, self).__init__(df)
        pass

    def get_train_test(self) -> (pd.DataFrame, pd.DataFrame):
        pass

    def get_data(self) -> pd.DataFrame:
        return self.df

    def _preprocess(self):
        self.create_apt_deposit_rank()
        self.create_apt_area_deposit_rank()

    def create_apt_deposit_rank(self):
        # latitude와 longitude가 같은 것끼리 groupby하여 deposit의 평균을 계산
        df = self.df
        grouped_df = (
            df.groupby(["latitude", "longitude"])["deposit"]
            .agg(["mean", "count"])
            .reset_index()
        )

        # 전체 deposit의 평균 계산 (NaN 제외)
        overall_mean = df["deposit"].mean()

        # NaN 그룹 처리: deposit이 모두 NaN인 그룹의 평균을 전체 평균으로 설정
        grouped_df["mean"] = grouped_df["mean"].fillna(overall_mean)

        # mean을 기준으로 정렬
        grouped_df = grouped_df.sort_values(by="mean", ascending=False)

        # apt_deposit_rank 부여
        grouped_df["apt_deposit_rank"] = range(1, len(grouped_df) + 1)

        # 원래 데이터프레임에 apt_deposit_rank 병합
        df = df.merge(
            grouped_df[["latitude", "longitude", "apt_deposit_rank"]],
            on=["latitude", "longitude"],
            how="left",
        )

        self.df = df

    def create_apt_area_deposit_rank(self):
        # latitude, longitude, area_m2로 그룹화하여 deposit의 평균을 계산
        df = self.df
        grouped_area_df = (
            df.groupby(["latitude", "longitude", "area_m2"])["deposit"]
            .agg(["mean", "count"])
            .reset_index()
        )

        # 전체 deposit의 평균 계산 (NaN 제외)
        overall_mean = df["deposit"].mean()

        # NaN 그룹 처리: deposit이 모두 NaN인 그룹의 평균을 전체 평균으로 설정
        grouped_area_df["mean"] = grouped_area_df["mean"].fillna(overall_mean)

        # mean을 기준으로 정렬
        grouped_area_df = grouped_area_df.sort_values(by="mean", ascending=False)

        # apt_area_deposit_rank 부여
        grouped_area_df["apt_area_deposit_rank"] = range(1, len(grouped_area_df) + 1)

        # 원래 데이터프레임에 apt_area_deposit_rank 병합
        df = df.merge(
            grouped_area_df[
                ["latitude", "longitude", "area_m2", "apt_area_deposit_rank"]
            ],
            on=["latitude", "longitude", "area_m2"],
            how="left",
        )

        self.df = df
