# 수도권 아파트 전세가 예측 프로젝트

## 프로젝트 구조

<details>
    <summary> 프로젝트 구조</summary>

```bash
├── src # AI 모델 학습을 위한 부분
│   ├── config # config.yaml 값 가져 오는 함수 제공
│   ├── model # AI 모델 src ex) Light GBM, XGBoost
│   └── pre_process # 모델 학습전 전처리
│   └── custom_wandb
│   └── plot 
│   └── server 
├── data #.gitignore
│   └── .csv #.gitignore
│     └── processed # 기타 csv 저장을 위한 저장소
|     └── raw # 원본 csv 저장을 위한 저장소
├── EDA # 개인 EDA 폴더
│   └── {팀원 명} 
│        ├──*.ipynb
├── app.py # 모델 학습을 위한 python 파일
├── config-sample.yaml # 하이퍼 파라미터 및 모델 & 서버 선택을 위한 설정 값
├── .gitignore
├── Readme.md
└── requirements.txt
```

</details>

<details>
    <summary> 라이브러리 버전</summary>

**Python 버전 : 3.12.5**

**Library 버전** - (requirements.txt)

```txt
numpy==2.1.1
pandas==2.2.3
lightgbm==4.4.0
scikit-learn==1.5.2
tqdm==4.66.4
xgboost==2.1.1
scipy==1.14.1
black==24.8.0
plotly==5.24.1
matplotlib==3.9.2
geopy==2.4.1
swifter
folium==0.17.0
pyproj==3.7.0
wandb==0.18.3
optuna==4.0.0
```

</details> 

## 목차
1. 프로젝트 소개
2. 팀 구성 및 역할
3. 절차 및 방법
4. EDA & Feature Engineering
5. Modeling
6. 최종 결과

## 1. 프로젝트 소개

* 프로젝트 기간 : 2024/09/30 ~ 2024/10/24

* 프로젝트 평가 기준 : 전세 실 거래 가격과 예측 가격의 MAE(Mean Absolute Error)

* 데이터 : upstage 대회에서 제공(아래 설명 O)

* 프로젝트 개요
> [upstage](https://stages.ai/)의 [수도권 아파트 전세가 예측 모델 대회](https://stages.ai/competitions/314/overview/description) 참가를 위한 프로젝트. <br>
약 180만 건의 2019/04\~2023/12월 수도권 아파트 전세 실거래가 데이터를 사용해, 약 15만 건의 2024/01\~2024/06월 전세 가격을 예측해야 한다.

## 2. 팀 구성 및 역할

| 이름   | 역할                                         |
|--------|----------------------------------------------|
| [김건율](https://github.com/ChoonB) | 팀장, EDA, 피처 엔지니어링, LGBM 모델링 및 앙상블 |
| [백우성](https://github.com/13aek) | EDA, 피처 엔지니어링, XGBoost 모델링          |
| [유대선](https://github.com/xenx96) | 프로젝트 설계, 모델링 자동화, EDA            |
| [이제준](https://github.com/passi3) | EDA, RF 모델링                               |
| [황태결](https://github.com/minari-c) | 공원, 학교 데이터 EDA                        |

## 3. 절차 및 방법

### 프로젝트 진행 과정
1. 도메인 지식을 위한 스터디 진행 후, 주어진 데이터에 대해서 토론.
2. 프로젝트를 위한 기본 구조 설립 및 코드 작성. (유대선)
3. 주제(금리, 지하철, 공원, 학교, 아파트)별로 각자 EDA 진행.
4. 각자 도출한 결과에 대해 공유하고 데이터 분석과 토론을 진행해 feature와 model 선택
5. Feature Engineering으로 필요한 feature 생성
6. 모델 훈련 및 하이퍼파라미터 튜닝
7. 최종 제출 선택

### 협업 방식
- Slack : 팀 간 실시간 커뮤니케이션, [Github 연동] 이슈 공유, 허들에서 실시간 소통 진행

- Zoom : 정기적인 회의와 토론을 위해 사용
- GitHub : 버전 관리와 코드 협업을 위해 사용. 각 팀원은 기능 단위로 이슈를 생성해 이슈 별 브랜치를 만들어 작업하고, Pull Request를 통해 코드 리뷰 후 병합하는 방식으로 진행
- GitHub Projects + Google Calendar : 팀원 간 일정 공유

## 4. EDA & Feature Engineering
> 우선 도메인 지식 스터디를 통해 팀원 모두가 데이터를 이해하고, 각자 주제 별 EDA를 진행한 뒤, 근거 있는 컬럼에 대해 Feature Engineering 진행.
(각자 EDA 한 내용은 github의 EDA-개인별 폴더에 정리)

### 데이터
- 데이터 셋 : train, test, subway, school, park
- train : 2019/04~2023/12월까지 약 180만 건의 전세 가격 실거래가 데이터와 계약 정보 데이터.
- test : 2024/01~06까지 약 15만 건의 전세 계약 정보 데이터.

<details>
    <summary>데이터 상세</summary>

| 컬럼 명 | 자료형 | 데이터 셋 | 설명 |
| --- | --- | --- | --- |
| index | int64 | train/test | 인덱스 |
| area_m2 | float64 | train/test | 면적 |
| contract_year_month | int64 | train/test | 계약년월 |
| contract_day | int64 | train/test | 계약일 |
| contract_type | int64 | train/test | 계약 유형(0: 신규, 1:갱신, 2:모름) |
| floor | int64 | train/test | 층수 |
| built_year | int64 | train/test | 건축 연도 |
| latitude | float64 | train/test/subway/school/park | 위도 |
| longitude | float64 | train/test/subway/school/park | 경도 |
| age | int64 | train/test | 건물나이 |
| deposit | float64 | train | **전세 실거래가(타겟 변수)** |
| year_month | int64 | interest | 연월 |
| interest_rate | float64 | interest | 이자율(연월에 해당하는 이자율) |
| schoolLevel | object | school | 초등학교, 중학교, 고등학교 여부 |
| area | float64 | park | 공원 면적 |

</details>


### 4-1. Apart data
- train의 deposit 분포는 오른쪽으로 치우친(right-skewed) 분포를 보인다. 그래서 EDA 과정에서는 로그변환을 사용해 분포를 맞춰 진행한다.
![](https://miniature-smelt-728.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F9480984b-9813-480f-8e62-ce0a81399f57%2Fa7076ed0-027f-4367-8aec-4170e7278754%2Fimage.png?table=block&id=2bdd8ec5-d0b9-4633-b331-06c2e4d4752c&spaceId=9480984b-9813-480f-8e62-ce0a81399f57&width=660&userId=&cache=v2)


- 상관계수를 확인할 땐, 크기(area_m2)의 영향을 받는 deposit을 바로 사용하지 않고, 부동산에서 많이 사용하는 개념인 평당 가격을 사용했다.
```python
df_train["area"] = (df_train["area_m2"] / 3.3).round(1)
df_train['area_price'] = 3.3*df_train['deposit'] / df_train['area']
```

- **Floor** : 층수 컬럼은 55층 정도가 넘어가면서 평균 가격이 확 뛰는 것으로 확인했다. 저층일 때보다 오히려 초 고층일때 전세가격 차이가 두드러졌다.
![](https://miniature-smelt-728.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F9480984b-9813-480f-8e62-ce0a81399f57%2Fe433dca6-b93a-422d-9672-74841e1636f3%2Fimage.png?table=block&id=4903355d-7741-4bd4-bdb5-0b1c6a6348db&spaceId=9480984b-9813-480f-8e62-ce0a81399f57&width=1420&userId=&cache=v2)

- **age** : 건물의 나이가 적다고 해서 무조건 전세가격이 높은 것은 아니다. 우하향 추세를 보이다 30~50년 사이에 전세 가격이 오르는 현상을 확인했다. 
![](https://miniature-smelt-728.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F9480984b-9813-480f-8e62-ce0a81399f57%2Fa4e58fbc-3c4b-4a28-926c-732527ff92aa%2Fimage.png?table=block&id=eadb9cc0-c31f-42f7-9152-dd70f905a253&spaceId=9480984b-9813-480f-8e62-ce0a81399f57&width=1420&userId=&cache=v2)

- **latitude-longtitude** : 위도, 경도가 같은 raw는 같은 아파트로 판단해 동일한 apt_idx를 부여했다.
- **contract_type** : 계약유형은 알수없음(2)이 많아 one-hot 인코딩 후 신규와 갱신 컬럼만 넣었다.
- **contract_year_month** : 관계된 컬럼은 연, 월, 일을 분해해서 월과 일을 사인, 코사인함수로 주기성을 줘봤지만, 오히려 계약연월일 단일컬럼으로 했을 때 보다 public score가 낮아 계약연월과 계약일은 합쳐서 단일 컬럼으로 사용했다.
- **area_m2** : 아파트의 크기는 별도의 변환 없이 바로 상관관계를 찍어봐도 0.52의 강한 상관관계를 보였다.
![](https://miniature-smelt-728.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F9480984b-9813-480f-8e62-ce0a81399f57%2Fa674bd77-5892-4d15-bd9a-b521ec1399f6%2Fimage.png?table=block&id=a0092e54-62ee-4644-af6f-d91e4e26462f&spaceId=9480984b-9813-480f-8e62-ce0a81399f57&width=770&userId=&cache=v2)

- **기본 컬럼** : train에 있는 기본 컬럼들을 로그 변환 후 평당가격과 측정한 상관계수는 area_m2를 제외하곤 절대값으로 0.17을 넘는 컬럼이 거의 없었다. 그러나 기본 컬럼들을 제외했을 때 보다 포함했을 때 MAE 결과가 더 잘나와 사용하기로 결정했다.
> XGBOOST : 기본 컬럼 제외 VAL-MAE:3944.77092 / 기본 컬럼 포함 VAL-MAE:3888.09750

### 4-2. subway, school, park data
> 부동산에선 ‘O세권’이란 단어를 사용한다. 일반적으로 아파트 근접 인프라가 잘 형성되어있으면 가격이 높아진다. 주어진 subway, school, park info 데이터의 위도, 경도를 활용해서 아파트 별 역세권, 학세권, 공세권을 판단할 수 있는 피처를 만들었다.

- 앞서 apt_idx 피처로 위도,경도가 같은 raw는 동일한 아파트로 판단 했기 때문에, apt_idx가 같으면 여기선 같은 피처 값을 가진다. 인프라(지하철, 학교, 공원)의 인덱스는 주어진 파일 순서대로 부여했다.
- 아래와 같은 방식으로 위도, 경도를 라디안으로 변환해 haversine 방식으로 BallTree를 만들어 아파트-인프라 거리와 그 인프라의 인덱스 번호를 피처로 생성했다.
```python
# 지구의 평균 반경 (킬로미터 단위, Haversine 공식에 필요)
EARTH_RADIUS_KM = 6371.0
# 아파트의 위도와 경도를 라디안으로 변환
temp_df_rad = np.radians(temp_df[["latitude", "longitude"]].values)
# 지하철 역의 위도와 경도를 라디안으로 변환
subway_info_rad = np.radians(subway_info[["latitude", "longitude"]].values)
# BallTree 생성 (Haversine 사용)
tree = BallTree(subway_info_rad, metric="haversine")
# 각 아파트에 대해 가장 가까운 지하철역과의 거리 찾기
distances, indices = tree.query(temp_df_rad, k=1)
# 거리 (라디안)를 미터 단위로 변환
temp_df["nearest_subway_distance"] = (
        distances.flatten() * EARTH_RADIUS_KM * 1000
)  # meters
# 가장 가까운 지하철역의 subway_idx 추출
temp_df["nearest_subway_idx"] = (
    subway_info["subway_idx"].iloc[indices.flatten()].values
)
```

- 상관관계를 확인해보기 위해 분포가 기울어진 평당 가격은 로그변환을 취했다. 가장 가까운 지하철 피처와 전세가격, 평당 전세가격과의 상관관계는 아래와 같다. 지하철역과 아파트가 가까울수록(음의 상관관계) 평당 전세가격이 상승하는 관계다.
![](https://miniature-smelt-728.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F9480984b-9813-480f-8e62-ce0a81399f57%2F2b4fac80-fbb2-4464-9841-0795348be29f%2Fimage.png?table=block&id=6df90435-212b-4c7f-9763-0e4b6b65a362&spaceId=9480984b-9813-480f-8e62-ce0a81399f57&width=1420&userId=&cache=v2)

- 거리별로 카테고리화 시켜서 평균 전세가격과 평균 평당가격을 살펴봐도 가까울수록 비싸다.
![](https://miniature-smelt-728.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F9480984b-9813-480f-8e62-ce0a81399f57%2F974eab85-7d26-44e7-8300-fbb65aa83682%2Fimage.png?table=block&id=5601504a-1f31-4c8e-9ad6-d5007f4ce32f&spaceId=9480984b-9813-480f-8e62-ce0a81399f57&width=2000&userId=&cache=v2)

- 공원의 경우도 대체적으로 지하철과 비슷한 경향을 보이고, 학교도 일정 거리까진 가까울수록 평당 가격이 비싼 관계를 보인다.
![](https://miniature-smelt-728.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F9480984b-9813-480f-8e62-ce0a81399f57%2Fb6160e2a-2ad3-4f56-bd48-bd6355e48f52%2Fimage.png?table=block&id=eb9cee0e-ac45-46fc-9a1b-2d7f50cce3de&spaceId=9480984b-9813-480f-8e62-ce0a81399f57&width=1420&userId=&cache=v2)

- 상관계수만 볼 때는 지하철과의 거리말고는 수치가 적어, 아래 4-4의 infra_count에서 새 피처를 추가로 만들어 사용했다.

| **feature** | **nearest_school_distance** | **nearest_park_distance** | **nearest_subway_distance** |
| --- | --- | --- | --- |
| corr | -0.059121 | -0.108378 | -0.410597 |

### 4-3. interest data

- 그래프 상으로는 금리와 전세 가격 사이에 반비례 관계가 존재한다고 추정할 수 있다. 하지만 실제 corr 값으로 확인해본 바, 전체 train dataset으로는 금리와 실거래가 사이의 관계가 있다고 보기 힘들다.
![](https://miniature-smelt-728.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F9480984b-9813-480f-8e62-ce0a81399f57%2Fde1a9838-3472-4e55-b74b-cd184848ec35%2Fimage.png?table=block&id=89209527-d814-43f1-aefb-d6369fe30c27&spaceId=9480984b-9813-480f-8e62-ce0a81399f57&width=1420&userId=&cache=v2)

- 평균 전세가 등락률과 금리와의 관계는 0.47로 평균 실거래가와 금리와의 지수보다는 상승한다. 또한 deposit의 중간 값을 통한 corr 값은 아래와 같다.
![](https://miniature-smelt-728.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F9480984b-9813-480f-8e62-ce0a81399f57%2Fb3008987-7557-493f-9864-95b6a06daa76%2Fimage.png?table=block&id=69c2cbb5-1921-41cf-8f70-a33aab301589&spaceId=9480984b-9813-480f-8e62-ce0a81399f57&width=1420&userId=&cache=v2)

| **corr** | **monthly_transaction** | **interest_rate** | **avg_deposit** | **deposit_diff** |
| --- | --- | --- | --- | --- |
| **monthly_transaction** | 1.000000 | 0.469879 | 0.455964 | -0.219402 |
| **interest_rate** | 0.469879 | 1.000000 | 0.130927 | -0.043196 |
| **avg_deposit** | 0.455964 | 0.130927 | 1.000000 | 0.129704 |
| **deposit_diff** | -0.219402 | -0.043196 | 0.129704 | 1.000000 |

- 시계열 데이터로 활용할 수 있는 데이터라 SARIMAX에 interest_rate를 외생 변수로 넣어서 평균 전세 가격 추이를 추정해 피처로 활용했지만, public score 상에서 결과가 좋지 않아 폐기했다.
![](https://miniature-smelt-728.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F9480984b-9813-480f-8e62-ce0a81399f57%2F3b848a7e-ef93-4cb7-ad12-37ce0a9f3a08%2Fimage.png?table=block&id=12968372-ae8c-8055-aba8-cda56e7e68fa&spaceId=9480984b-9813-480f-8e62-ce0a81399f57&width=1420&userId=&cache=v2)

- 하지만 특정 모델(XGBoost)에서 이자와 관련된 피처(interest_rate, diff_interest_rate)를 오히려 제외할 때 public score(MAE)가 600이상 감소했고, val-mae와 test-mae간의 격차도 완화되었다. 
![](https://miniature-smelt-728.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F9480984b-9813-480f-8e62-ce0a81399f57%2F79431a3a-26ff-49b2-adc7-709198b3d94c%2Fimage.png?table=block&id=b54ac4ec-4a75-4ca6-bd00-b1753435a278&spaceId=9480984b-9813-480f-8e62-ce0a81399f57&width=880&userId=&cache=v2)

- 결론적으로, 이자 관련 피처가 결과를 왜곡한다고 판단하여 기본 interest_rate만 사용하거나 LGBM같은 특정 모델에서는 아예 피처에서 제외했다.

### 4-4. Feature Creation
#### recent_deposit
> 일반적으로 부동산 가격을 볼 때 가장 많이 보는 지표는 ‘최근 거래가’이다. 그래서 apt_idx와 area_m2가 같은 가장 최근 raw의 deposit을 찾아 최근 거래가 피처를 만들었다.

- 최근 거래가를 바로 타겟에 넣고 결측치는 train deposit의 평균으로 채워 test에 제출해봤을 때 public score MAE가 4644가 나와, 그 자체로 상당히 좋은 지표로 사용될 여지가 보였다.
- deposit에서 파생된 데이터이므로 상관계수는 0.927089로 상당히 높았다.
- 하지만 이 피처를 만드는 과정에서 같은 아파트임에도 가격 차이가 크게 나는 계약들을 발견했다. 지나치게 낮은 가격의군집을 공공 임대 계약으로 가정하고 피처에서 제외하기 위해 mean or median값의 50%를 threshold로 삼아서 그 값 이하는 최근 거래가에서 제외했다.
![](https://miniature-smelt-728.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F9480984b-9813-480f-8e62-ce0a81399f57%2Fc2ea92c1-5a8c-4159-b66b-9448ef0cec42%2Fimage.png?table=block&id=12968372-ae8c-807e-9203-e0ef5e71590a&spaceId=9480984b-9813-480f-8e62-ce0a81399f57&width=1420&userId=&cache=v2)

#### apt_deposit_rank
> 같은 apt_idx를 가진 raw끼리 groupby해서 평균 전세가격을 구하고, 서로 비교해 랭킹을 매겨놓은 피처. 

- test에만 존재하는 apt_idx는 랭킹을 추정할 수 없으므로 평균 전세 가격을 넣어 결측치를 보충했다.
- apt_idx와 area_m2(면적)까지 같은 raw끼리 groupby해서 동일한 방식으로 apt_area_deposit_rank 피처도 생성했다.

#### grid_deposit
> 외부 행정동 데이터를 사용할 수 없기 때문에 주어진 위도, 경도를 기준으로 일정 크기의 격자(grid)로 나누어 행정동을 대체해서 근처 아파트들 끼리 묶어 활용했다. 이를 통해 test에 있는 아파트들이 어떤 구역에 속하는 지 파악이 가능해진다.

- 위도, 경도 좌표를 UTM 좌표계로 변환해 utm_x, utm_y를 3km로 구역을 나눠 490개의 grid로 재편했다. 각 grid 구획의 평균 전세 가격을 대표값으로 사용해 grid_deposit 피처로 생성했다. 로그 변환된 평당 전세가격과 상관계수는 0.720374로 측정되었다.
![](https://miniature-smelt-728.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F9480984b-9813-480f-8e62-ce0a81399f57%2F2f2ccf1f-7561-4ae4-94e2-f6028bf1a56d%2Fimage.png?table=block&id=12968372-ae8c-8040-88dc-c0262fa9179b&spaceId=9480984b-9813-480f-8e62-ce0a81399f57&width=1420&userId=&cache=v2)

#### infra_count
> 4-2의 infra distance 피처만으로는 부족해 새롭게 생성했다. 각 grid에 속하는 subway, park, school의 수를 count해서 피처로 활용했다. 로그 변환된 평당 전세 가격과 상관계수는 0.493523(지하철), 0.226171(학교), 0.213145(공원)으로 나왔다.

## 5. Modeling
- 초기 모델 : XGBoost & LightGBM
> XGBoost는 레벨 단위로 트리를 생성하고, LightGBM은 리프 단위로 트리를 생성하는 트리기반의 의사결정 알고리즘 모델이다. 두 모델 다 그라디언트 부스팅 기반으로, 대규모 데이터 셋에서 빠른 학습 속도와 효율적인 메모리 활용으로 사용하기 편하다. 그리고 비선형적이고 복잡한 관계도 잘 파악하며, Feature Importance를 제공해 영향력이 큰 피처를 찾기 쉬워서 사용을 결정했다.

- 평가 지표 : MAE
> 대회 측정 지표로 활용하는 MAE(Mean Absolute Error, 평균 절대 오차)를 동일하게 사용했다.

- Validation 전략 : K-fold CV (k=5)
> 부동산 데이터의 경우 지역 별 특성이 중요해 누락되는 데이터가 있으면 안된다고 생각했다. 모든 데이터가 훈련과 검증을 참여할 수 있는 k-fold CV 방식을 사용했다. k값은 여러 값을 넣어보고 계산 효율과 성능 평가에 있어 적당한 값이 5라고 판단했다.

- 하이퍼 파라미터 조정
> Optuna를 통해 베이지안 방식으로 하이퍼파라미터를 찾아 해당 값을 사용해봤으나 train과 eval MAE 격차가 700이상 벌어졌고, public score MAE가 더 크게 나와 과적합 이슈로 인해 포기했다. 그래서 랜덤 서치 방식으로 하이퍼파라미터를 조정했다.

- 추가적으로 시도해본 모델
1. **GRU** :  시계열 모델인 GRU의 사용을 위해 기본적인 코드를 짜서 모델링을 시도해 보았지만, 프로젝트 막바지에 시도해 시계열 모델을 제대로 적용하는 데 시간적 한계가 있어 중단했다.
2. **Random Forest Regressor** : 다수의 결정 트리를 생성해 예측 결과를 다수결로 결정해 이상치에 강하다고 알려져있는 랜덤포레스트를 이용하면 MAE 방식의 채점 기준에 적합할 것으로 예상되었으나 기본 학습에 걸리는 시간과 튜닝을 위한 시간이 너무 많이 소요되고, 다른 모델보다 낮은 성능으로 확인되기 때문에 앙상블 이후 배제했다.

- **Stacking Ensemble**
> XGBoost, LightGBM, RandomForest, Gradient Boosting, ElasticNet Regressor 모델을 기본 모델로 선정해서 독립적으로 학습시켜 validation(random_state=42, test_size=0.2)을 만든다. Linear Regression(or LGBM)를 메타 모델로 선정해 기본모델에서 만들어진 validation을 학습시켜 최종 결과를 stacking 방식으로 도출한다.
하지만 학습시간이 오래 걸리고, 단순 평균이나 단독 모델보다 public score에서 우위를 찾을 수 없어 프로젝트에서 배제했다. (Stacking MAE : 3789.6573 vs LGBM MAE :3730.6235)

- **Weighted Average Ensemble**
> 최종적으로 살아남은 XGBoost와 LightGBM 모델의 예측 값들을 가중 평균해 제출을 시도했다. public score가 높은 쪽에 가중치를 더 줘서 평균을 내보니 단독 모델보다 MAE가 더 낮게 나와 스태킹 방식이 아닌  가중 평균 앙상블 방식을 계속 사용했다.

## 6. 최종 결과
1. 아래의 서로 다른 하이퍼파라미터와 피처를 가진 XGBoost와 LightGBM 4가지 모델의 타겟 값(deposit)을 가중 평균해서 final 생성(0.125+0.125+0.25+0.5)
- public score : 3483.0271 / private score : 4279.3377

2. V9 데이터셋을 적용한 두 모델 LightGBM 25% + XGBoost 75%를 섞어 생성
- public score : 3506.4542 / private score : 4307.9871

<details>
    <summary>V9 데이터셋</summary>

```python
columns = ['contract_date_numeric', 'area_m2', 'floor', 'built_year', 'latitude',
       'longitude', 'age', 'contract_0', 'contract_1', 'deposit', 'apt_idx',
       'area', 'grid_deposit', 'apt_deposit_rank', 'apt_area_deposit_rank',
       'recent_deposit', 'nearest_park_distance', 'nearest_park_idx',
       'park_area', 'nearest_school_distance', 'nearest_school_idx',
       'nearest_subway_distance', 'nearest_subway_idx', 'park_count',
       'school_count', 'subway_count']
```
</details>

<details>
    <summary>최종 모델 하이퍼 파라미터</summary>

1. LGBM 1 HyperParameters
```json
{
  "objective": "regression",
  "metric": "mae",
  "boosting_type": "gbdt",
  "num_leaves": 200,
  "learning_rate": 0.01,
  "feature_fraction": 0.8,
  "bagging_fraction": 0.7,
  "bagging_freq": 1,
  "num_boost_round": 15000,
  "early_stopping_rounds": 100,
  "gpu_platform_id": 0,
  "gpu_device_id": 0,
}
```
2. XGBoost 1 HyperParameters
```json
 {
  "objective": "reg:absoluteerror",
  "eval_metric": "mae",
  "max_depth": 10,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "learning_rate": 0.01,
  "num_boost_round": 20000,
  "early_stopping_rounds": 100,
  "verbose_eval": 100,
  "device": "cuda"
 }
```
3. LGBM 2 HyperParameters
```json
{
  "objective": "regression",
  "metric": "mae",
  "boosting_type": "gbdt",
  "num_leaves": 64,
  "learning_rate": 0.05,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "feature_fraction": 0.8,
  "lambda_l2": 0.1,
  "bagging_fraction": 0.7,
  "bagging_freq": 1,
  "num_boost_round": 50000,
  "early_stopping_rounds": 1000,
  "n_jobs": -1
}
```
4. XGBoost 2 HyperParameters
```json
{
  "objective": "reg:absoluteerror",
  "eval_metric": "mae",
  "max_depth": 6,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "learning_rate": 0.05,
  "min_child_weight": 10,
  "reg_lambda": 0.1,
  "reg_alpha": 0,
  "num_boost_round": 50000,
  "early_stopping_rounds": 1000,
  "verbose_eval": 1000,
  "n_jobs": -1,
  "gamma": 0,
}
```
</details>
