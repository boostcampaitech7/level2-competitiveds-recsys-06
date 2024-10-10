import pandas as pd

from typing import List
from src.pre_procecss.interface import PreProcessInterface


class FeatureDuplication(PreProcessInterface):
    def __init__(self, df: pd.DataFrame):
        super(FeatureDuplication, self).__init__(df)
        pass

    def get_train_test(self) -> (pd.DataFrame, pd.DataFrame):
        pass

    def get_data(self) -> pd.DataFrame:
        return self.df

    def _preprocess(self):
        self.df = self.df.drop(columns=["index"]).drop_duplicates(keep="first")
