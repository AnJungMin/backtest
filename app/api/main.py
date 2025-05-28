from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.predict import router as predict_router
import os

app = FastAPI()

# ✅ static 폴더 등록 (히트맵 이미지 접근용)
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영 환경에서는 프론트엔드 도메인으로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 예측 라우터 등록
app.include_router(predict_router, prefix="/api", tags=["prediction"])

# __main__에서 실행할 경우: uvicorn 실행
if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=port, reload=True)
