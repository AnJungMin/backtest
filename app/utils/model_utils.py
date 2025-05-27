import os
import torch
from torchvision import transforms
from PIL import Image
from app.model.model import get_model  # 반드시 모델 구조와 같아야 함

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join("app", "model_weight", "final_best_model_v2.pth")

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model():
    model = get_model().to(DEVICE)
    # PyTorch 2.6.0에선 weights_only=True 명시!
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
    class_labels = ["정상", "이상"]  # 클래스명 직접 확인
    return {
        "class": class_labels[pred_class],
        "confidence": f"{confidence:.2f}%",
        "class_index": pred_class
    }
