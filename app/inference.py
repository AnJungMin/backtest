import sys
import os
from PIL import Image
from app.utils.model_utils import load_model, predict_image

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

    try:
        # 모델 한 번만 로드 (최적화)
        model = load_model()
    except Exception as e:
        print(f"[ERROR] 모델 로드 실패: {e}")
        sys.exit(1)

    try:
        # 이미지 로드 및 변환
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] 이미지 열기 실패: {e}")
        sys.exit(1)

    try:
        # 예측 수행
        result = predict_image(image, model)
    except Exception as e:
        print(f"[ERROR] 예측 중 오류 발생: {e}")
        sys.exit(1)

    print(result)

if __name__ == "__main__":
    main()
