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
        print("[ERROR] 모델이 로드되지 않았습니다.")
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다.")

    try:
        # ✅ 파일 읽기 및 이미지 변환
        print("[INFO] 파일 수신 및 이미지 변환 시도")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        print("[INFO] 이미지 변환 성공")
    except Exception:
        print("[ERROR] 유효하지 않은 이미지 파일")
        raise HTTPException(status_code=400, detail="유효한 이미지 파일이 아닙니다.")

    try:
        # ✅ 예측 및 히트맵 생성
        print("[INFO] 예측 및 히트맵 생성 시작")
        result = predict_image(image, model)
        print(f"[INFO] 예측 결과: {result}")
        heatmap = generate_heatmap(image, model)
        print("[INFO] 히트맵 생성 성공")

        # ✅ static 경로 생성 및 저장
        static_dir = "static"
        os.makedirs(static_dir, exist_ok=True)
        heatmap_filename = f"heatmap_{uuid.uuid4().hex}.jpg"
        heatmap_path = os.path.join(static_dir, heatmap_filename)
        heatmap.save(heatmap_path)
        print(f"[INFO] 히트맵 저장 완료: {heatmap_path}")

        # ✅ 응답 포맷 수정 (프론트가 기대하는 구조에 맞게)
        response = {
            "class": result["class"],
            "confidence": result["confidence"],
            "class_index": result["class_index"],
            "heatmap_url": f"/static/{heatmap_filename}"
        }
        print(f"[INFO] 응답 반환: {response}")
        return response

    except Exception as e:
        print(f"[ERROR] 예측 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")
