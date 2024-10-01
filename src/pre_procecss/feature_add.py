import pandas as pd

from typing import List
from src.pre_procecss.interface import PreProcessInterface


class FeatureAddition(PreProcessInterface):
    def __init__(self, df: pd.DataFrame):
        pass

    def get_train_test(self) -> (pd.DataFrame, pd.DataFrame):
        pass

    def get_data(self) -> pd.DataFrame:
        return self.df

    def _preprocess(self):
        self.create_apt_idx()
        self.create_area_price()

    def create_area_price(self):
        df = self.df
        df["area"] = (df["area_m2"] / 3.3).round(1)
        df["area_price"] = df["deposit"] / df["area_m2"]
        df["area_m2_price"] = df["deposit"] / df["area_m2"]
        self.df = df

    def create_apt_idx(self):
        df: pd.DataFrame = self.df
        # create column
        df["apt_idx"] = 0

        # create lon+lat mapper column
        lon = df["longitude"].astype(str)
        lat = df["latitude"].astype(str)
        df["lon_lat"] = lon + lat

        # Create apt_idx data-frame
        lon_lat = pd.DataFrame(df["lon_lat"].unique(), columns=["lon_lat"])
        lon_lat.reset_index(inplace=True, drop=False)
        lon_lat.set_index("lon_lat", inplace=True)
        lon_lat.columns = ["apt_idx"]
        # Add Apt_idx
        df["apt_idx"] = df["lon_lat"].map(lon_lat["apt_idx"])

        # Drop lon+lat mapper column
        df.drop(columns=["lon_lat"], inplace=True)

        self.df = df
