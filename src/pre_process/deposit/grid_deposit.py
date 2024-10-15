import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Proj
from pyproj import Transformer

from typing import List
from src.pre_process.interface import PreProcessInterface

class GridFeature(PreProcessInterface):
    def __init__(self, df: pd.DataFrame, grid_size: int = 3000, **kwargs):
        """
        :param df: 입력 데이터프레임
        :param grid_size: 격자 크기, 기본값은 3000 (미터 단위)

        위도와 경도 기반으로 UTM 좌표 계산 후 grid_x, grid_y 생성
        grid_x, grid_y 기준으로 deposit의 평균을 계산
        location_df와 grid_deposit_mean을 병합하여 grid_mean 생성
        grid_mean을 기반으로 새로운 grid_id 생성
        "epsg:4326", "epsg:32652": 한국 지역의 UTM Zone 52
        """
        self.grid_size = grid_size 
        self.transformer = Transformer.from_crs("epsg:4326", "epsg:32652")
        super().__init__(df, **kwargs)

    def get_train_test(self) -> (pd.DataFrame, pd.DataFrame):
        pass

    def get_data(self) -> pd.DataFrame:
        return self.df

    def _preprocess(self):
        self.df['utm_x'], self.df['utm_y'] = zip(
            *self.df.apply(lambda row: self.latlon_to_utm(row['latitude'], row['longitude']), axis=1))

        self.df['grid_x'] = (self.df['utm_x'] // self.grid_size).astype(int)
        self.df['grid_y'] = (self.df['utm_y'] // self.grid_size).astype(int)

        grid_deposit_mean = self.df.groupby(['grid_x', 'grid_y'])['deposit'].mean().reset_index()

        self.df = pd.merge(self.df, grid_deposit_mean, on=['grid_x', 'grid_y'], 
                           how='left', suffixes=('', '_grid_mean'))
        
        self.df['grid_id'] = self.df['deposit_grid_mean']

        self.df.drop(['utm_x', 'utm_y', 'grid_x', 'grid_y', 'deposit_grid_mean'], 
                     axis=1, inplace=True)

    def latlon_to_utm(self, lat, lon):
        """
        위도와 경도를 UTM 좌표로 변환
        """
        utm_x, utm_y = self.transformer.transform(lat, lon)
        return utm_x, utm_y
        