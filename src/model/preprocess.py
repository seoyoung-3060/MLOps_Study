'''
데이터 전처리 함수들
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

from src.configs.config_loader import load_config

config = load_config()
COLUMNS = config["columns"]
PREPROCESSING_CONFIG = config["preprocessing_config"]
CATEGORY_MAPPINGS = config["category_mappings"]

def winsorize_series(s, lower_quantile=None, upper_quantile=None):
    """
    윈저라이징(상/하한 범위로 자르기)
    Args:
        s: pandas Series
        lower_val: 하한 분위수 (None일 경우 config에서 가져옴)
        upper_val: 상한 분위수 (None일 경우 config에서 가져옴)
    Return:
        윈저라이징된 Series
    """
    if lower_quantile is None or upper_quantile is None:
        lower_quantile = PREPROCESSING_CONFIG["winsorize"]["lower_quantile"]
        upper_quantile = PREPROCESSING_CONFIG["winsorize"]["upper_quantile"]

    lower_val = s.quantile(lower_quantile)
    upper_val = s.quantile(upper_quantile)
    return s.clip(lower_val, upper_val)


# IQR 기반 이상치 처리
def iqr_capping(df, col, factor=None):
    '''
    IQR 기반 이상치 처리
    Args:
        df: dataframe
        col: columns
        factor: IQR 배수
    '''
    if factor is None:
        factor = PREPROCESSING_CONFIG["iqr_capping"]["factor"]
    
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + factor * IQR
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])


def create_derived_feature(df):
    '''
    파생변수 생성
    Args:
        df: dataframe
    Return:
        파생변수 추가된 df
    '''
    df[COLUMNS["연체 없음"]] = (df["마지막 연체 이후 경과 개월 수"] == 0).astype(int)
    return df

def categorical_mapping(df):
    '''
    범주형 데이터 mapping 처리
    Args:
        df: dataframe
    Return:
        범주형 데이터 처리된 df
    '''
    df_categorical_mapping = df.copy()
    for col, mapping in CATEGORY_MAPPINGS.items():
        if col in df_categorical_mapping.columns:
            df_categorical_mapping[col] = df_categorical_mapping[col].map(mapping)
    return df_categorical_mapping


def iqr_capping(df, columns):
    '''
    IQR 캐핑 적용
    Args:
        df: dataframe
        col: columns
    Return:
        iqr 적용된 df    
    '''
    df_iqr = df.copy()
    factor = PREPROCESSING_CONFIG["iqr_capping"]["factor"]
    for col in columns:
        if col in df_iqr.columns:
            iqr_capping(df_iqr, factor)
    return df_iqr

def log_transform(df):
    '''
    로그 변환
    Args:
        df: dataframe
    Return:
        로그 변환된 df
    '''
    df_transformed = df.copy()
    log_cols = PREPROCESSING_CONFIG["log_columns"]
    for col in log_cols:
        if col in df_transformed.columns:
            df_transformed[col] = winsorize_series(df_transformed[col])
            df_transformed[col] = np.log1p(df_transformed[col])
    return df_transformed


def iqr_transform(df):
    '''
    iqr capping 진행
    Args:
        df: dataframe
    Return:
        iqr capping된 df
    '''
    df_iqr = df.copy()
    iqr_cols = PREPROCESSING_CONFIG["iqr_columns"]
    for col in iqr_cols:
        if col in df_iqr.columns:
            df_iqr[col] = iqr_capping(df_iqr, col)
    return df_iqr


def preprocess_data(train_df, test_df):
    '''
    전체 데이터 파이프라인
    Args:
        train_df : 훈련 데이터, test_df : 테스트 데이터
    Return:
        전처리된 train_df, test_df, test_uid
    '''
    # UID 처리
    # test_uid = test_df[[COLUMNS["uid"]]]
    train_df = train_df.drop(COLUMNS["uid"])
    test_df = test_df.drop(COLUMNS["uid"])
    
    # 파생변수 생성
    train_df = create_derived_feature(train_df)
    test_df = create_derived_feature(test_df)
    
    # 범주형 데이터 처리
    train_df = categorical_mapping(train_df)
    test_df = categorical_mapping(test_df)
        
    # 로그변환
    train_df = log_transform(train_df)
    test_df = log_transform(test_df)
    
    # IQR capping
    train_df = iqr_transform(train_df)
    test_df = iqr_transform(test_df)
    
    # KNN Imputer
    knn_imputer = KNNImputer(n_neighbors=PREPROCESSING_CONFIG["knn_imputer"]["n_neighbors"])
    train_df = pd.DataFrame(knn_imputer.fit_transform(train_df), columns=train_df.columns)
    test_df = pd.DataFrame(knn_imputer.fit_transform(test_df), columns=test_df.columns)    
    
    # StandardScaler
    scaler = StandardScaler()
    train_df = pd.DataFrame(scaler.fit_transform(train_df), columns=train_df.columns)
    test_df = pd.DataFrame(scaler.fit_transform(test_df), columns=test_df.columns)
    
    return train_df, test_df
