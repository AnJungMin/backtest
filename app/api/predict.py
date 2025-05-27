from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image
import io
from app.utils.model_utils import load_model, predict_image

router = APIRouter()

try:
    model = load_model()  # 서버 시작 시 1회 모델 로드, 재사용
except Exception as e:
    print(f"[ERROR] 모델 로드 실패: {e}")
    model = None

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="유효한 이미지 파일이 아닙니다.")

    try:
        result = predict_image(image, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {e}")

    return result
