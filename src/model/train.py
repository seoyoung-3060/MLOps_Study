# src/model/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import os

from src.model.model import build_model
from src.model.preprocess import preprocess_data

import mlflow
from mlflow import sklearn

def train_model(config):
    '''
    config.yaml 기반으로 전체 학습 파이프라인 실행
    '''
    print("config 기반 모델 학습 시작")
    
    train_path = config["data_paths"]["train"]
    test_path = config["data_paths"]["test"]
    target_col = config["columns"]["target"]

    # 데이터 불러오기
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 데이터 전처리
    train_df, test_df = preprocess_data(train_df, test_df)

    # X,y 나누기
    X = train_df.drop(columns=target_col)
    y = train_df[target_col]

    # 데이터 나누기
    split_config = config["model_config"]["train_test_split"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=split_config["test_size"],
        random_state=split_config["random_state"]
    )

    # 모델 학습
    model = build_model()
    model.fit(X_train, y_train)

    # 평가
    print("✅ Train Score:", model.score(X_train, y_train))
    print("✅ Test Score:", model.score(X_test, y_test))

    # 모델 저장하기
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "../models/sgd_model.pkl")
    print("모델을 저장합니다 : sgd_model.pkl")