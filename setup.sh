echo "----MLOps_Study 프로젝트 환경설정----"
echo "Python 가상환경을 설치하겠습니까? (y/n)"
read -r creat_venv

if ["$creat_venv" == "y"] || ["$creat_venv" == "Y"]; then
    echo "가상환경 생성중.."
    python -m venv mlops_env

    if [["$OSTYPE" == "msys"]]; then
        echo "Linux/Mac 가상환경을 활성화합니다."
        source mlops_env/bin/activate

    else
        echo "Windows에서 가상환경을 활성화합니다."
        source mlops_env/Scrips/activate

    fi
fi

echo "필요한 패키지 설치 중..."
pip install --upgrade pip
pip install -r requirements.txt

echo "데이터 파일 확인중..."
if [! -f "train.csv"]; then
    echo "Warning : train.csv 파일이 없습니다."
fi
if [! -f "test.csv"]; then
    echo "Warning : test.csv 파일이 없습니다."
fi

echo "🎉 환경 설정이 완료되었습니다!"
echo "다음 명령어로 모델 훈련을 시작할 수 있습니다."
echo "python -m src.model.train"