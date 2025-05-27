from fastapi import APIRouter, File, UploadFile
from PIL import Image
import io
import torch
from torchvision import transforms
import os
from app.model.model import get_model  # 반드시 모델 구조 함수 import

router = APIRouter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ✅ state_dict 방식으로 로드 (get_model 사용 O)
def load_model():
    model_path = os.path.join("app", "model_weight", "final_best_model_v2.pth")
    model = get_model().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

def predict_image(image: Image.Image, model):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return int(predicted.item())

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    pred = predict_image(image, model)
    return {"prediction": pred}
