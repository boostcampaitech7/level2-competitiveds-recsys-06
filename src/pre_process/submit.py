import pandas as pd

from src.config import get_config
from src.pre_process.interface import PreProcessInterface


class Submit(PreProcessInterface):
    def __init__(self, df: pd.DataFrame, **kwargs):
        super().__init__(df, **kwargs)
        self.df = df
        # self.type_feature =

    def get_train_test(self) -> (pd.DataFrame, pd.DataFrame):
        pass

    def _preprocess(self):
        type_feature = get_config().get("data").get("type_feature")
        for c in self.df.columns:
            if (self.df[c].dtype == "object" and c != type_feature) or c == "index":
                self.df.drop(columns=[c], inplace=True, errors="ignore")

    def get_data(self) -> pd.DataFrame:
        return self.df
