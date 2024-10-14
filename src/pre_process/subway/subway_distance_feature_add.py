import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

from typing import List
from src.pre_process.interface import PreProcessInterface
from src.pre_process.subway.subwayInfo_feature_add import SubwayInfoFeatureAddition


class SubwayDistanceFeatureAddition(PreProcessInterface):
    """
    train, test 데이터에 subway_info를 활용해서 지하철 관련 피처를 추가하는 클래스
    df에는 train과 test를 concat해서 feature_add.py를 통해서 apt_idx를 추가시킨 데이터프레임을 넣어야 한다.
    subway_info에는 전처리 안된 subwayInfo.csv를 넣으면 SubwayInfoFeatureAddition 클래스를 통해 자동으로 전처리 시킨다.

    반경 1km 데이터는 넣고, 반경 500m 데이터는 제외.
    추가되는 피처:
    - nearest_subway_distance: 가장 가까운 지하철역까지의 거리 (미터 단위)
    - nearest_subway_idx: 가장 가까운 지하철역의 subway_idx
    - num_subway_within_1km: 반경 1km 내의 지하철역 개수
    - category_interchange_within_1km: 반경 1km 내에 환승역 존재 여부 카테고리
    (0: 지하철역 없음, 1: 지하철역 1개이상(환승역x), 2: 지하철역 1개이상(환승역 포함))
    - num_subway_within_500m: 반경 500m 내의 지하철역 개수
    - category_interchange_within_500m: 반경 500m 내에 환승역 존재 여부 카테고리
    (0: 지하철역 없음, 1: 지하철역 1개이상(환승역x), 2: 지하철역 1개이상(환승역 포함))

    """

    def __init__(self, df: pd.DataFrame, subway_info: pd.DataFrame, **kwargs):
        self.subway_info: pd.DataFrame = SubwayInfoFeatureAddition(
            subway_info
        ).get_data()
        super(SubwayDistanceFeatureAddition, self).__init__(df, **kwargs)
        pass

    def get_train_test(self) -> (pd.DataFrame, pd.DataFrame):
        pass

    def get_data(self) -> pd.DataFrame:
        return self.df

    def _preprocess(self):
        self.add_subway_features()
        pass

    def add_subway_features(self):
        df = self.df
        subway_info = self.subway_info
        # apt_idx를 기준으로 아파트별로 하나씩만 뽑은 데이터프레임
        temp_df = df.sort_values(by=["apt_idx", "contract_year_month"]).drop_duplicates(
            subset="apt_idx", keep="first"
        )

        # 지구의 평균 반경 (킬로미터 단위, Haversine 공식에 필요)
        EARTH_RADIUS_KM = 6371.0

        # 아파트의 위도와 경도를 라디안으로 변환
        temp_df_rad = np.radians(temp_df[["latitude", "longitude"]].values)
        # 지하철 역의 위도와 경도를 라디안으로 변환
        subway_info_rad = np.radians(subway_info[["latitude", "longitude"]].values)

        # BallTree 생성 (Haversine 거리 메트릭 사용)
        tree = BallTree(subway_info_rad, metric="haversine")

        # 각 아파트에 대해 가장 가까운 지하철역과 그 거리 찾기
        distances, indices = tree.query(temp_df_rad, k=1)

        # 거리 (라디안)를 미터 단위로 변환
        temp_df["nearest_subway_distance"] = (
            distances.flatten() * EARTH_RADIUS_KM * 1000
        )  # meters

        # 가장 가까운 지하철역의 subway_idx 추출
        temp_df["nearest_subway_idx"] = (
            subway_info["subway_idx"].iloc[indices.flatten()].values
        )

        # 반경 1km을 라디안으로 변환
        radius = 1 / EARTH_RADIUS_KM

        # 각 아파트에 대해 반경 1km 내의 지하철역 인덱스 찾기
        indices_within_1km = tree.query_radius(temp_df_rad, r=radius)

        # 반경 1km 내의 지하철역 개수
        temp_df["num_subway_within_1km"] = [len(ind) for ind in indices_within_1km]

        # (추가적인 EDA 진행되면 추가하고 일단 제외)
        # # 반경 1km 내의 지하철역 subway_idx 리스트
        # temp_df["list_subway_idx_within_1km"] = [
        #     subway_info["subway_idx"].iloc[ind].tolist() for ind in indices_within_1km
        # ]

        """
        반경 1km 내에 Interchange_station이 2 이상인 지하철역 존재 여부를 기반으로 'category_interchange_within_1km' 열 생성
        조건:
        0: 지하철역 없음
        1: 지하철역 1개 이상, 환승역 없음
        2: 지하철역 1개 이상, 환승역 포함
        """

        temp_df["category_interchange_within_1km"] = [
            (
                0
                if len(ind) == 0
                else (
                    2 if subway_info["Interchange_station"].iloc[ind].ge(2).any() else 1
                )
            )
            for ind in indices_within_1km
        ]

        # 반경 500m을 라디안으로 변환
        radius_500m = 0.5 / EARTH_RADIUS_KM

        # 각 아파트에 대해 반경 500m 내의 지하철역 인덱스 찾기
        indices_within_500m = tree.query_radius(temp_df_rad, r=radius_500m)

        # 반경 500m 내의 지하철역 개수
        temp_df["num_subway_within_500m"] = [len(ind) for ind in indices_within_500m]

        # 각 아파트에 대해 반경 500m 내의 Interchange_station 개수 계산
        temp_df["category_interchange_within_500m"] = [
            (
                0
                if len(ind) == 0
                else (
                    2 if subway_info["Interchange_station"].iloc[ind].ge(2).any() else 1
                )
            )
            for ind in indices_within_500m
        ]

        # 'category_interchange_within_1km', '~500m' 컬럼을 0과 1로 변환
        temp_df["category_interchange_within_1km"] = temp_df[
            "category_interchange_within_1km"
        ].astype(int)

        temp_df["category_interchange_within_500m"] = temp_df[
            "category_interchange_within_500m"
        ].astype(int)

        # 'nearest_subway_distance' 컬럼을 반올림해서 정수형으로 변환
        temp_df["nearest_subway_distance"] = (
            temp_df["nearest_subway_distance"].round(1).astype(int)
        )

        # temp_df와 df를 일치하는 데이터끼리 매칭시켜서 temp_df의 열을 df에 추가
        df = df.merge(
            temp_df[
                [
                    "apt_idx",
                    "nearest_subway_distance",
                    "nearest_subway_idx",
                    "num_subway_within_1km",
                    "category_interchange_within_1km",
                    "num_subway_within_500m",
                    "category_interchange_within_500m",
                ]
            ],
            on=["apt_idx"],
            how="left",
        )

        self.df = df
