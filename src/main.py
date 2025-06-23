# main.py

from src.configs.config_loader import load_config
from src.model.train import train_model


def main():
    print("🚀 MLOps 파이프라인 실행 시작!")
    config = load_config()
    train_model(config)

if __name__ == "__main__":
    main()
