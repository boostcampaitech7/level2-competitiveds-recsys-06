# 수도권 집값 예측 프로젝트

## 1. 프로젝트 구조

<details>
    <summary> 프로젝트 구조</summary>

[GoogleDrive(DATA 저장 경로)](https://drive.google.com/drive/folders/1V87hu50-TKT2XQSWAIif8YzoVygJdc1l?hl=ko)

```bash
├── src # AI 모델 학습을 위한 부분
│   ├── config # config.yaml 값 가져 오는 함수 제공
│   ├── model # AI 모델 src ex) Light GBM, XGBoost
│   └── pre_process # 모델 학습전 전처리
├── data #.gitignore
│   └── .csv #.gitignore
│     └── processed # 기타 csv 저장을 위한 저장소
├── EDA # 개인 EDA 폴더
│   └── {팀원 명} 
│        ├──*.ipynb
├── app.py # 모델 학습을 위한 python 파일
├── config-sample.yaml # 하이퍼 파라미터 및 모델 & 서버 선택을 위한 설정 값
├── .gitignore
├── Readme.md
└── requirements.txt
```

시작전 해당 경로를 등록해주세요!

```python
# src 경로 추가 - src를 불러오기위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(""))))
print(sys.path[-1])
```

</details>

<details>
    <summary> 라이브러리 버전</summary>

**Python 버전 : 3.12.5**

**Library 버전** - (requirements.txt)

```txtpandas==2.2.2
numpy==1.26.4
lightgbm==4.4.0
scikit-learn==1.5.0
tqdm==4.66.4
xgboost==2.0.3
plotly==5.22.0
scipy==1.11.4
black
plotly
```

</details> 

## 목차

1. 프로젝트 소개

## 1. 프로젝트 소개

* 프로젝트 기간 : 2024/09/30 ~ 2024/10/25

* Boostcamp RecSys 6조
  팀원 : [김건율](https://github.com/ChoonB), [백우성](https://github.com/13aek), [유대선](https://github.com/xenx96), [이제준](https://github.com/passi3), [황태결](https://github.com/minari-c)

* 프로젝트 목표
