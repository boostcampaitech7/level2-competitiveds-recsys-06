from abc import abstractmethod, ABC

import pandas as pd


class ModelInterface(ABC):
    @abstractmethod
    def __init__(self, x_train: pd.DataFrame, y_train: pd.DataFrame, config: any):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self, test_df: pd.DataFrame):
        pass

    @abstractmethod
    def train_validation(self) -> None:
        pass

    @abstractmethod
    def export_model(self, dir_path: str):
        pass
