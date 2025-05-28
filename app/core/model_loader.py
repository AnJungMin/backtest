import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image

# ✅ 장치 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] DEVICE 설정: {DEVICE}")

# ✅ 모델 경로 (상대경로 유지)
MODEL_PATH = os.path.join("app", "model_weight", "final_best_model_v2.pth")
print(f"[INFO] 모델 경로: {MODEL_PATH}")

# ✅ 이미지 전처리
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ EfficientNet-B0 모델 구조 정의
def get_model(num_classes=4):
    print("[INFO] EfficientNet-B0 모델 구조 생성")
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    print("[INFO] Output Linear Layer 교체 완료")
    return model

# ✅ 모델 로딩
def load_model():
    print("[INFO] 모델 로딩 시작")
    model = get_model(num_classes=4).to(DEVICE)
    print("[INFO] 모델 구조 to(DEVICE) 완료")
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    print("[INFO] state_dict 로드 완료")
    model.load_state_dict(state_dict)
    print("[INFO] 모델 가중치 load_state_dict 완료")
    model.eval()
    print("[INFO] 모델 eval() 모드 전환 완료")
    return model

# ✅ 이미지 예측
def predict_image(image: Image.Image, model: nn.Module):
    print("[INFO] 예측 이미지 전처리 시작")
    image = data_transforms(image).unsqueeze(0).to(DEVICE)
    print(f"[INFO] 이미지 shape: {image.shape}")
    with torch.no_grad():
        print("[INFO] 모델 추론 시작")
        output = model(image)
        probs = torch.softmax(output, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item() * 100
        print(f"[INFO] 추론 결과 - pred_class: {pred_class}, confidence: {confidence:.2f}%")
    class_labels = ["양호", "경증", "중등도", "중증"]
    result = {
        "class": class_labels[pred_class],
        "confidence": f"{confidence:.2f}%",
        "class_index": pred_class
    }
    print(f"[INFO] 반환 결과: {result}")
    return result
