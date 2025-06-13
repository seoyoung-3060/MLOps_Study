# src/model/model.py

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.configs.config_loader import load_config

def build_model():
    config = load_config()
    model_config = config["model_config"]["sgd"]

    pipeline = make_pipeline(
        StandardScaler(),
        SGDClassifier(**model_config)
    )
    
    return pipeline
