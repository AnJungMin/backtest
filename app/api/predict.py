from fastapi import APIRouter, File, UploadFile
from PIL import Image
import io
from app.utils.model_utils import load_model, predict_image

router = APIRouter()
model = load_model()  # 서버 시작시 1회만 모델 로드, API 요청마다 재사용

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    result = predict_image(image, model)
    return result
