import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join("app", "model_weight", "final_best_model_v2.pth")  # 실제 경로로 수정하세요

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def get_model(num_classes=6):
    # EfficientNet-b0 모델 생성 (pretrained=False)
    model = efficientnet_b0(weights=None)
    # 마지막 fully connected layer를 클래스 수에 맞게 수정
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def load_model():
    model = get_model(num_classes=6).to(DEVICE)  # 클래스 수를 학습 때 맞게 설정하세요
    # PyTorch 2.6 이상: weights_only=True 로 state_dict만 로드
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_image(image, model):
    image = data_transforms(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item() * 100
    class_labels = ["정상", "이상", "경증", "중등도", "중증", "심각"]  # 예시: 클래스명 실제에 맞게 수정하세요
    return {
        "class": class_labels[pred_class],
        "confidence": f"{confidence:.2f}%",
        "class_index": pred_class
    }
