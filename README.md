# MLOPS_STUDY 

채무 불이행 예측을 위한 SGD(Stochastic Gradient Descent) 분류 모델을 모듈화하는 프로젝트입니다.

## 프로젝트 구조

```
MLOPS_STUDY/
├── src/
│   ├── __init__.py               # 패키지 인식용
│   ├── configs/
│   │   ├── __init__.py
│   │   ├── config_loader.py      # config 로딩
│   │   └── config.yaml           # 설정 관리
│   └── model/
│       ├── __init__.py
│       ├── model.py              # 모델 구성 및 파이프라인 정의
│       ├── preprocess.py         # 데이터 전처리
│       └── train.py              # 모델 학습 및 제출 파일 생성
├── data/                         # 데이터 파일 (train.csv, test.csv)
├── models/                       # 훈련된 모델 저장소 (자동 생성됨)
├── README.md
├── requirements.txt
└── setup.sh

```

## 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 데이터 준비
- `train.csv`와 `test.csv` 파일을 프로젝트 data 디렉토리에 배치

### 3. 모델 훈련
```bash
python -m src.model.train
```

## 기능

### 데이터 전처리
- **파생 변수 생성**: 연체 없음 여부 플래그
- **범주형 데이터 인코딩**: 대출 상환 기간, 대출 목적, 근속 연수, 주거 형태
- **이상치 처리**: Winsorization (1%~99% 범위로 클리핑)
- **IQR 캐핑**: 연간 소득 이상치 제거
- **로그 변환**: 신용액, 부채액, 대출 잔액
- **결측치 처리**: KNN Imputation (k=10)
- **정규화**: StandardScaler 적용

### 모델
- **알고리즘**: SGD Classifier
- **손실 함수**: Log Loss (로지스틱 회귀)
- **정규화**: L2 penalty
- **최대 반복**: 1000회
- **파이프라인 구성**: StandardScaler + SGDClassifier

### 출력
- 학습/검증 점수 출력
- models/sgd_model.pkl에 모델 저장

## 설정 커스터마이징

`src/configs/config.py`에서 다음 항목들을 수정할 수 있습니다:

- 데이터 파일 경로
- 전처리 설정 (IQR, 분위수, 로그변환 대상 등)
- 모델 하이퍼파라미터 (loss, penalty, max_iter, 등)
- 컬럼명 및 매핑 정보


## 확장 가능성

이 모듈화된 구조를 통해 다음과 같은 확장이 용이합니다:
- 새로운 전처리 방법 추가
- 다른 모델 알고리즘 적용
- 하이퍼파라미터 튜닝 자동화
- MLOps 파이프라인 연동