import os
import torch
from torchvision import transforms
from PIL import Image
import timm  # timm EfficientNet 사용 시

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join("app", "model_weight", "final_best_model_v2.pth")

# ⭐ 신뢰 글로벌 등록 (timm EfficientNet 기준)
torch.serialization.add_safe_globals({'EfficientNet': timm.models.efficientnet.EfficientNet})

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model():
    # 전체 모델 객체 복원
    model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = model.to(DEVICE)
    model.eval()
    return model

def predict_image(image, model):
    image = data_transforms(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item() * 100
    class_labels = ["정상", "이상"]  # 실제 클래스명에 맞게 조정
    return {
        "class": class_labels[pred_class],
        "confidence": f"{confidence:.2f}%",
        "class_index": pred_class
    }
