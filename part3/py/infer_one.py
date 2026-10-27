import sys  # گرفتن آرگومان‌های خط فرمان
import torch  # لود مدل و اجرای شبکه
from PIL import Image  # خواندن تصویر

# تلاش برای وارد کردن کلاس مدل (اگر اسم کمی فرق داشت هم هندل شود)
try:
    from train_mri import SmallCNN as ModelClass
except Exception:
    from train_mri import SmalCNN as ModelClass  # اگر اسم کلاس بدون l بوده باشد

def preprocess_image(image_path: str, size: int = 128) -> torch.Tensor:
    img = Image.open(image_path)  # باز کردن فایل تصویر
    img = img.convert("RGB")  # تبدیل به سه کانال (مدل 3 کاناله است)
    img = img.resize((size, size))  # تغییر اندازه مثل زمان آموزش

    x = torch.tensor(list(img.get_flattened_data()), dtype=torch.float32)  # گرفتن پیکسل‌ها
    x = x.view(size, size, 3)  # شکل‌دهی به H×W×C
    x = x / 255.0  # نرمال‌سازی به بازه 0..1
    x = x.permute(2, 0, 1)  # تبدیل به C×H×W
    x = x.unsqueeze(0)  # افزودن بعد batch → 1×3×128×128

    return x

def main():
    if len(sys.argv) < 3:
        print("Usage: python infer_one.py model.pt image.jpg")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    class_names = ["glioma", "meningioma", "notumor", "pituitary"]

    device = torch.device("cpu")
    model = ModelClass(num_classes=4).to(device)
    model.eval()

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    x = preprocess_image(image_path, 128).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    pred_id = int(pred.item())
    conf = float(confidence.item())
    label = class_names[pred_id] if 0 <= pred_id < len(class_names) else "unknown"

    print(f"pred={pred_id} label={label} conf={conf:.4f}")

if __name__ == "__main__":
    main()
