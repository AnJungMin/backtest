import numpy as np
import cv2
import torch
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
from app.core.model_loader import DEVICE, data_transforms  # 기존 전처리 재사용

def generate_heatmap(image: Image.Image, model: torch.nn.Module) -> Image.Image:
    model.eval()

    # 1. 이미지 전처리
    input_tensor = data_transforms(image).unsqueeze(0).to(DEVICE)

    # 2. 마지막 conv layer feature 추출 설정 (EfficientNetB0 기준)
    return_nodes = {"features.6.2.conv": "feat"}  # 해당 레이어 이름 정확해야 함
    feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

    # 3. forward pass & feature map 추출
    with torch.no_grad():
        features = feature_extractor(input_tensor)
        output = model(input_tensor)

    # 4. 예측 클래스에 대한 backprop
    pred_class = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, pred_class].backward(retain_graph=True)

    # 5. 수동 gradient 추적을 하지 못하므로 fallback 예시
    # 실제 프로젝트에서는 conv output에 hook 달아 gradient 추적 필요함

    # 6. CAM 계산용 임시 처리 (주의: gradient 없이 weights 평균 사용)
    activation = features["feat"].squeeze(0).cpu().numpy()
    cam = np.mean(activation, axis=0)  # gradient 가중치 대신 평균 사용
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    # 7. 시각화 및 원본 이미지 위에 합성
    heatmap = cv2.resize(cam, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original_img = image.resize((224, 224))
    original_np = np.array(original_img)

    if original_np.ndim == 2 or original_np.shape[2] == 1:
        original_np = cv2.cvtColor(original_np, cv2.COLOR_GRAY2RGB)

    overlay = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)

    return Image.fromarray(overlay)
