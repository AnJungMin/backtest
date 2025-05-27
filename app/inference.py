import sys
import os
from PIL import Image
from app.model_utils import load_model, predict_image

def main():
    # 이미지 경로 입력받기 (인자 없으면 test.jpg)
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "test.jpg"

    # 파일 존재 여부 체크
    if not os.path.isfile(image_path):
        print(f"[ERROR] 이미지 파일이 존재하지 않습니다: {image_path}")
        sys.exit(1)

    # 이미지 로드 및 예측
    image = Image.open(image_path).convert("RGB")
    model = load_model()
    result = predict_image(image, model)
    print(result)

if __name__ == "__main__":
    main()
