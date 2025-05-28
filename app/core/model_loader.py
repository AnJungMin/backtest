import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image

# ✅ 장치 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 모델 경로 (상대경로 유지)
MODEL_PATH = os.path.join("app", "model_weight", "final_best_model_v2.pth")

# ✅ 이미지 전처리
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ EfficientNet-B0 모델 구조 정의
def get_model(num_classes=4):
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

# ✅ 모델 로딩
def load_model():
    model = get_model(num_classes=4).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ✅ 이미지 예측
def predict_image(image: Image.Image, model: nn.Module):
    image = data_transforms(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item() * 100
    class_labels = ["양호", "경증", "중등도", "중증"]
    return {
        "class": class_labels[pred_class],
        "confidence": f"{confidence:.2f}%",
        "class_index": pred_class
    }
