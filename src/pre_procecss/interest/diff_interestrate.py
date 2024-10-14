import pandas as pd

from src.pre_process.interface import PreProcessInterface

class DiffInterestRateAdder(PreProcessInterface):
    """
    두 개의 데이터프레임을 인자로 받아서 이전 달 대비 이자율 변화량을 계산하고 병합합니다.
    매개변수로 받은 df에서 datetime으로 구성된 열을 확인하고 datetime 형식의 열이 없을 시 "year_month"를 포함하는 열을 datetime 형식으로 변환합니다.
    해당 열을 기준으로 병합한 후, 오름차순으로 정렬합니다.
    "interestRate" 칼럼이 존재하는지 확인하여 이전 달 대비 변화량을 의미하는 "diff_interestRate" 칼럼을 추가합니다.
    """
    def __init__(self, df1: pd.DataFrame, df2: pd.DataFrame):
        self.df1 = df1
        self.df2 = df2
        self.merged_df = None
        self._preprocess()

    def get_train_test(self) -> (pd.DataFrame, pd.DataFrame):
        pass

    def _preprocess(self):
        # df1과 df2에서 datetime 형식의 첫 번째 칼럼을 찾아 반환
        datetime_col_df1 = self._get_datetime_column(self.df1)
        datetime_col_df2 = self._get_datetime_column(self.df2)

        # datetime 형식으로 변환
        self.df1[datetime_col_df1] = pd.to_datetime(self.df1[datetime_col_df1], format="%Y%m")
        self.df2[datetime_col_df2] = pd.to_datetime(self.df2[datetime_col_df2], format="%Y%m")
        # 두 데이터프레임 병합
        self.merged_df = pd.merge(self.df1, self.df2, left_on=datetime_col_df1, right_on=datetime_col_df2, how="inner")
        # datetime 칼럼을 기준으로 오름차순 정렬
        self.merged_df.sort_values(by=datetime_col_df1, inplace=True)
        self.merged_df.reset_index(drop=True, inplace=True)
        # interest_rate 칼럼이 존재하는지 확인하고 diff_interestRate 칼럼 추가
        if "interest_rate" in self.merged_df.columns:
            self.merged_df["diff_interest_rate"] = self.merged_df["interest_rate"].diff()
        else:
            raise ValueError("병합된 데이터프레임에 'interest_rate' 칼럼이 존재하지 않습니다.")

    def get_data(self) -> pd.DataFrame:
        return self.merged_df
    
    def _get_datetime_column(self, df: pd.DataFrame):
        # 데이터프레임에서 datetime 형식의 칼럼을 확인하고 없을 경우 year_month를 포함하는 칼럼을 반환
        datetime_columns = df.select_dtypes(include=["datetime"]).columns
        if len(datetime_columns) == 0:
            year_month_columns = [col for col in df.columns if "year_month" in col]
            if len(year_month_columns) != 0:
                return year_month_columns[0]
            else:
                raise ValueError("데이터프레임에 datetime 형식의 칼럼이나 'year_month'를 포함하는 칼럼이 존재하지 않습니다.")
        
        return datetime_columns[0]