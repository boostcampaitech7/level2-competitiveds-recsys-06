from abc import abstractmethod, ABC

import pandas as pd


class PreProcessInterface(ABC):
    @abstractmethod
    def __init__(self, df: pd.DataFrame):
        self.df: pd.DataFrame = df
        self._preprocess()
        pass

    @abstractmethod
    def get_train_test(self) -> (pd.DataFrame, pd.DataFrame):
        pass

    @abstractmethod
    def _preprocess(self):
        pass

    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        pass
