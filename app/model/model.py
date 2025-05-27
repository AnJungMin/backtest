import timm
import torch.nn as nn

def get_model():
    """
    EfficientNet B0 (timm), num_classes=2로 이진 분류 모델 반환
    """
    model = timm.create_model('tf_efficientnet_b0_ns', pretrained=True, num_classes=2)
    return model

# 테스트 (직접 실행 시)
if __name__ == '__main__':
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")  # torch.Size([1, 2])
