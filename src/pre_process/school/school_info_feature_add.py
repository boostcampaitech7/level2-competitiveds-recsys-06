import pandas as pd
import numpy as np

from src.pre_process.interface import PreProcessInterface
from sklearn.neighbors import BallTree


class SchoolInfoFeatureAdd(PreProcessInterface):
    """
    schoolinfo.csv를 불러와 입력 df에 컬럼을 추가해주는 클래스
    idx: 기본으로 주어진 순서
    """

    def __init__(self, df: pd.DataFrame, school_df: pd.DataFrame,  **kwargs):
        self.school_df = school_df
        super(SchoolInfoFeatureAdd, self).__init__(df, **kwargs)
        pass

    def get_train_test(self) -> (pd.DataFrame, pd.DataFrame):
        pass

    def get_data(self) -> pd.DataFrame:
        return self.df

    def _preprocess(self):
        self.add_school_feature()

    def add_school_feature(self, interval: float=1):
        '''
            temp_df에 추가할 컬럼을 합친 뒤 df와 merge하여 최종적으로 원하는 컬럼의 내용을 df에 추가한다.

            추가 컬럼
            [
                'apt_idx'
                for l in ['elementary', 'middle', 'high']:
                    col_names.extend([
                        f'nearest_{l}_school_distance'
                        , f'nearest_{l}_school_within_{interval_str}'
                        , f'has_{l}_school_within_{interval_str}'
                ])
            ]
            여기서 school level에 따라 지표를 추출했다.

            interval_str:
            f'{interval}km' if interval >= 1. else f'{interval * 1000}m'
            -> km 혹은 m 단위를 선택하여 str로 변환해준다.
            -> 추가할 컬럼명에 사용한다.

            radius:
            산정구간(기본0.5km) / EARTH_RADIUS_KM
            -> 산정구간은 아파트를 기준점으로한다
        '''
        interval_str = f'{interval}km' if interval >= 1. else f'{interval * 1000}m'
        EARTH_RADIUS_KM = 6371.0
        radius = interval / EARTH_RADIUS_KM
        df = self.df
        school_df = self.school_df

        # apt_idx 기준 가장 최신 하나만 -> 각 아파트마다 가장 최근 거래 자료만
        temp_df = df.sort_values(by=['apt_idx', 'contract_year_month']).drop_duplicates(
            subset='apt_idx', keep='first'
        )
        temp_df_rad = np.radians(temp_df[['latitude', 'longitude']].values)
        school_sorted = school_df.sort_values(by=['latitude', 'longitude'])

        for l in ['elementary', 'middle', 'high']:
            # 위도 경도를 라디안으로 변환
            school_sorted_rad = np.radians(school_sorted[school_sorted['schoolLevel'] == l][['latitude', 'longitude']].values)

            tree = BallTree(school_sorted_rad, metric='haversine')

            distances, indices = tree.query(temp_df_rad, k=1)

            # 거리(라디안) -> 거리(km) 변환
            temp_df[f'nearest_{l}_school_distance'] = distances.flatten() * EARTH_RADIUS_KM * 1000

            indices_within_distance = tree.query_radius(temp_df_rad, r=radius)

            temp_df[f'nearest_{l}_school_within_{interval_str}'] = [len(ind) for ind in indices_within_distance]
            temp_df[f'has_{l}_school_within_{interval_str}'] = [len(p) > 0 for p in indices_within_distance]

        col_names = []
        for l in ['elementary', 'middle', 'high']:
            col_names.extend([
                f'nearest_{l}_school_distance'
                , f'nearest_{l}_school_within_{interval_str}'
                , f'has_{l}_school_within_{interval_str}'
            ])
        col_names.append('apt_idx')

        self.df = df.merge(
            temp_df[col_names]
            , on=['apt_idx']
            , how='left'
        )
