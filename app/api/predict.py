from fastapi import APIRouter, File, UploadFile
from PIL import Image
import io
import torch
from torchvision import transforms
import os

router = APIRouter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ✅ 전체 모델 객체로 로드 (get_model 사용 X)
def load_model():
    model_path = os.path.join("app", "model_weight", "final_best_model_v2.pth")
    model = torch.load(model_path, map_location=device)
    model = model.to(device)
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
