#! /bin/bash

# 로그 디렉토리 생성
mkdir -p logs

# 가상환경 활성화
. .venv/bin/activate

# 백그라운드로 실행하고 로그 저장
nohup streamlit run src/app.py > logs/app.log 2>&1 &

# 실행된 프로세스 ID 출력
echo "애플리케이션이 백그라운드에서 실행되었습니다."
echo "프로세스 ID: $!"
echo "로그 확인: tail -f logs/app.log"