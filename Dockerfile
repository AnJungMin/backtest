FROM python:3.11-slim

WORKDIR /app

# ✅ OpenCV가 사용하는 시스템 라이브러리 설치
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ✅ 소스 복사 및 패키지 설치
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# ✅ 서버 실행
ENTRYPOINT ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
