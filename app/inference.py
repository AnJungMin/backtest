import os
import torch
from torchvision import transforms
from PIL import Image
from app.model.model import get_model  # EfficientNet 구조 반환
# from app.train.config import DEVICE  # DEVICE를 내부에서 직접 정의

# 디바이스 정의 (내부에서 처리)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 경로 (final_best_model_v2.pth)
MODEL_PATH = os.path.join("app", "model_weight", "final_best_model_v2.pth")

# 전처리 (튜닝.ipynb 코드와 동일)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 모델 로드
def load_model():
    model = get_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# 예측 함수
def predict_image(image=None, image_path=None, model=None):
    # 이미지 로드
    if image is None and image_path is not None:
        image = Image.open(image_path).convert("RGB")
    elif image is None:
        raise ValueError("Image or image_path must be provided.")

    # 전처리
    input_tensor = data_transforms(image).unsqueeze(0).to(DEVICE)

    # 추론
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item() * 100

    # 클래스 이름은 실제 학습 데이터에 맞게 조정!
    class_labels = ["정상", "이상"]  # 예시(이진 분류라면)
    result = {
        "class": class_labels[pred_class],
        "confidence": f"{confidence:.2f}%",
        "class_index": pred_class
    }

    return result

# 테스트용 코드 (직접 실행 시)
if __name__ == "__main__":
    model = load_model()
    image_path = "test.jpg"  # 예시
    result = predict_image(image_path=image_path, model=model)
    print(result)
