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
    print("[INFO] 모델 로딩 시작")
    model = load_model()
    print("[INFO] 모델 로딩 완료")
except Exception as e:
    print(f"[ERROR] 모델 로드 실패: {e}")
    model = None

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="모델 로드 실패")

    # 1) 이미지 읽기
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="유효한 이미지 파일이 아닙니다.")

    # 2) 예측 + 히트맵 생성
    try:
        result = predict_image(image, model)
        heatmap_img: Image.Image = generate_heatmap(image, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 오류: {e}")

    # 3) static/heatmaps 디렉토리에 저장
    #    → main.py 에서 `app.mount("/static", StaticFiles(directory="app/static"))` 했다고 가정
    heatmaps_dir = os.path.join("app", "static", "heatmaps")
    os.makedirs(heatmaps_dir, exist_ok=True)

    heatmap_filename = f"{uuid.uuid4().hex}.png"
    save_path = os.path.join(heatmaps_dir, heatmap_filename)
    heatmap_img.save(save_path)  # PIL 이미지 저장

    # 4) 클라이언트에 반환할 URL 경로
    heatmap_url = f"/static/heatmaps/{heatmap_filename}"

    return {
        "class":        result["class"],
        "confidence":   result["confidence"],
        "class_index":  result["class_index"],
        "heatmap_url":  heatmap_url
    }