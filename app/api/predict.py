from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image
import io
import os
import uuid

from app.core.model_loader import load_model, predict_image
from app.core.gradcam import generate_heatmap

router = APIRouter()

# ✅ 모델 미리 로드
try:
    model = load_model()
except Exception as e:
    print(f"[ERROR] 모델 로드 실패: {e}")
    model = None

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다.")

    try:
        # ✅ 파일 읽기 및 이미지 변환
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="유효한 이미지 파일이 아닙니다.")

    try:
        # ✅ 예측 및 히트맵 생성
        result = predict_image(image, model)
        heatmap = generate_heatmap(image, model)

        # ✅ static 경로 생성 및 저장
        static_dir = "static"
        os.makedirs(static_dir, exist_ok=True)  # 폴더 없을 경우 생성
        heatmap_filename = f"heatmap_{uuid.uuid4().hex}.jpg"
        heatmap_path = os.path.join(static_dir, heatmap_filename)
        heatmap.save(heatmap_path)

        # ✅ 응답 반환
        return {
            "result": result,
            "heatmap_url": f"/static/{heatmap_filename}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")
