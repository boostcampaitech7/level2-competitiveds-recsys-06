import pandas as pd

from typing import List
from src.pre_process.interface import PreProcessInterface


class FillDeposit(PreProcessInterface):
    def __init__(self, df: pd.DataFrame):
        super(FillDeposit, self).__init__(df)
        pass

    def get_train_test(self) -> (pd.DataFrame, pd.DataFrame):
        pass

    def get_data(self) -> pd.DataFrame:
        return self.df

    def _preprocess(self):
        self.fill_test_deposit_from_train()
        self.revert_5_percent_increased_values()

    def fill_test_deposit_from_train(self):
        # 1. train 데이터만 필터링
        train_data = self.df[self.df["_type"] == "train"]

        # 2. test 데이터만 필터링
        test_data = self.df[self.df["_type"] == "test"]

        # 3. 전체 train 데이터의 deposit 평균값 계산
        overall_train_mean = train_data["deposit"].mean()

        # 4. test 데이터 내에서 NaN이 있는 부분만 처리
        for i, row in test_data[test_data["deposit"].isna()].iterrows():

            # 같은 아파트 그룹과 같은 면적을 가진 train 데이터의 deposit 값을 찾음
            same_group_and_area = train_data[
                (train_data["apt_idx"] == row["apt_idx"])
                & (train_data["area_m2"] == row["area_m2"])
                ]

            if len(same_group_and_area) > 0:
                # 5. 같은 면적을 가진 아파트의 deposit 값에서 5% 인상된 값으로 채움
                self.df.at[i, "deposit"] = same_group_and_area["deposit"].mean() * 1.05
                self.df.at[i, "is_increased"] = True
            else:
                # 6. 같은 apt_idx만 있는 경우, 그 그룹의 deposit 평균값으로 채움
                same_group = train_data[train_data["apt_idx"] == row["apt_idx"]]
                if len(same_group) > 0:
                    group_mean = same_group["deposit"].mean()
                    self.df.at[i, "deposit"] = group_mean
                    self.df.at[i, "is_increased"] = False
                else:
                    # 7. apt_idx가 없으면 전체 train 데이터의 평균 deposit 값으로 채움
                    self.df.at[i, "deposit"] = overall_train_mean
                    self.df.at[i, "is_increased"] = False

        return self.df

        # 5% 인상된 값을 복원하는 함수 (is_increased가 True인 값만 복원)

    def revert_5_percent_increased_values(self):
        is_increased_index = self.df["is_increased"] == True

        # 5% 인상된 값만 1.05로 나눠 원래 값으로 복원
        self.df.loc[is_increased_index, "deposit"] = (
                self.df.loc[is_increased_index, "deposit"] / 1.05
        )
        return self.df
