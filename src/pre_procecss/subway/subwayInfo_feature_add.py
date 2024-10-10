import pandas as pd

from typing import List
from src.pre_procecss.interface import PreProcessInterface


class SubwayInfoFeatureAddition(PreProcessInterface):
    """
    subwayInfo.csv에 주어진 순서대로 subway_idx를 부여하고 환승역 카테고리 컬럼을 추가하는 클래스
    """

    def __init__(self, df: pd.DataFrame):
        super(SubwayInfoFeatureAddition, self).__init__(df)
        pass

    def get_train_test(self) -> (pd.DataFrame, pd.DataFrame):
        pass

    def get_data(self) -> pd.DataFrame:
        return self.df

    def _preprocess(self):
        self.create_subway_idx()
        self.create_subway_Interchange_station_category()

    def create_subway_idx(self):
        # subway_idx 컬럼을 추가하고 인덱스를 부여
        self.df["subway_idx"] = self.df.index

    def create_subway_Interchange_station_category(self):
        # 환승역 카테고리 컬럼을 추가
        subway_info = self.df

        # Interchange_station 컬럼 만들어서 위도/경도가 중복인 지하철역 개수 표시(자기자신 포함)
        # 즉, 환승역이 아니면 1이고, 역 2개가 겹치면 2. 그 이상은 겹치는 횟수만큼 증가.
        subway_info["Interchange_station"] = subway_info.groupby(
            ["latitude", "longitude"]
        )["latitude"].transform("count")

        # 환승역은 subway_idx가 가장 낮은 하나만 남기고 제거
        subway_info = subway_info.loc[
            subway_info.groupby(["latitude", "longitude"])["subway_idx"].idxmin()
        ]
        self.df = subway_info
