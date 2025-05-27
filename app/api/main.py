from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.predict import router as predict_router

app = FastAPI()

# CORS 설정 (배포 시에는 allow_origins에 프론트엔드 주소 넣기!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 예: ["https://fronttest-ajbj.onrender.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 예측 라우터 등록 (예: POST /api/predict)
app.include_router(predict_router, prefix="/api", tags=["prediction"])

# __main__에서 uvicorn 서버 실행 코드
if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)
