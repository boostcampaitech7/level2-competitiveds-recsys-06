import pandas as pd

from src.pre_process.interface import PreProcessInterface

class DiffInterestRateAdder(PreProcessInterface):
    """
    두 개의 데이터프레임을 인자로 받아서 이전 달 대비 이자율 변화량을 계산하고 병합합니다.
    매개변수로 받은 df에서 datetime으로 구성된 열을 확인하고, datetime 형식의 열이 없을 시 "year_month"를 포함하는 열을 datetime 형식으로 변환합니다.
    해당 열을 기준으로 병합한 후, 오름차순으로 정렬합니다.
    'interest_rate' 칼럼이 존재하는지 확인하여 이전 달 대비 변화량을 의미하는 'diff_interest_rate' 칼럼을 추가합니다.
    """
    def __init__(self, df: pd.DataFrame, df_interest: pd.DataFrame):
        self.df_interest = df_interest
        super(DiffInterestRateAdder, self).__init__(df)

    def get_train_test(self) -> (pd.DataFrame, pd.DataFrame):
        pass

    def get_data(self) -> pd.DataFrame:
        return self.merged_df
    
    def _preprocess(self):
        self._find_datetime_column()

    def _find_datetime_column(self):
        # df과 df_interest에서 datetime 형식의 첫 번째 칼럼을 찾아 반환
        datetime_col_df = self._get_datetime_column(self.df)
        datetime_col_df_interest = self._get_datetime_column(self.df_interest)

        # datetime 형식으로 변환
        self.df[datetime_col_df] = pd.to_datetime(self.df[datetime_col_df], format="%Y%m")
        self.df_interest[datetime_col_df_interest] = pd.to_datetime(self.df_interest[datetime_col_df_interest], format="%Y%m")
        
        # interest_rate 칼럼이 존재하는지 확인하고 diff_interest_rate 계산
        if 'interest_rate' in self.df_interest.columns:
            self.df_interest.sort_values(by=datetime_col_df_interest, inplace=True)
            self.df_interest['diff_interest_rate'] = self.df_interest['interest_rate'].diff()
        
        # 두 데이터프레임 병합
        self.merged_df = pd.merge(self.df, self.df_interest, left_on=datetime_col_df, right_on=datetime_col_df_interest, how="inner")
        
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