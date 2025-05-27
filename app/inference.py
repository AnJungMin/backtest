import sys
from PIL import Image
from app.model_utils import load_model, predict_image

if __name__ == "__main__":
    model = load_model()
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    image = Image.open(image_path).convert("RGB")
    result = predict_image(image, model)
    print(result)
