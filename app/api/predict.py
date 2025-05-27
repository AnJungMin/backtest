from fastapi import APIRouter, File, UploadFile
from PIL import Image
import io
from app.model_utils import load_model, predict_image

router = APIRouter()
model = load_model()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    result = predict_image(image, model)
    return result
