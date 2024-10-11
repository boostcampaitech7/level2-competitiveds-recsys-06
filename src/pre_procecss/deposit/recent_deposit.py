import pandas as pd

from typing import List
from src.pre_procecss.interface import PreProcessInterface

from tqdm import tqdm
import swifter

tqdm.pandas()


class RecentDepositFeatureAddition(PreProcessInterface):
    """
    df=train+test
    df에서 최근 거래된 (apt_idx / floor[없을경우 최근 거래]) 전세가 Feature를 추가하는 방법
    약 소요 시간 1시간~2시간 내외
    """

    def __init__(self, df: pd.DataFrame):
        super(RecentDepositFeatureAddition, self).__init__(df)
        pass

    def get_train_test(self) -> (pd.DataFrame, pd.DataFrame):
        pass

    def get_data(self) -> pd.DataFrame:
        return self.df

    def _preprocess(self):
        self._add_recent_deposit()

    def _add_recent_deposit(self):
        df: pd.DataFrame = self.df

        def match_recent_deposit(raw_df, row):
            before_df = raw_df.loc[raw_df["apt_idx"] == row["apt_idx"]]
            before_df = before_df.loc[before_df["contract_ymd"] < row["contract_ymd"]]
            before_df = before_df.loc[before_df["area_m2"] == row["area_m2"]]
            before_df = before_df.loc[before_df["deposit"] != -999.0]
            before_deposit = before_df["deposit"]
            if len(before_deposit) == 0:
                row["recent_deposit"] = -999
                return row

            floor_deposit = before_df.loc[before_df["floor"] == row["floor"]]["deposit"]
            if len(floor_deposit) == 0:
                row["recent_deposit"] = before_deposit.iloc[len(before_deposit) - 1]
            else:
                row["recent_deposit"] = floor_deposit.iloc[len(floor_deposit) - 1]
            return row

        df.sort_values(by=["contract_ymd", "apt_idx"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.reset_index(drop=False, inplace=True)
        result = df.swifter.apply(
            lambda x: match_recent_deposit(df.iloc[: x["index"]], x), axis=1
        )

        self.df = result
