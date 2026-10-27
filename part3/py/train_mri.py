import os  # کار با مسیرها
import time  # زمان‌گیری
import random  # رندوم
import numpy as np  # آرایه‌ها
from PIL import Image  # خواندن عکس
from tqdm import tqdm  # نوار پیشرفت

import torch  # هسته پای‌تورچ
import torch.nn as nn  # لایه‌ها
import torch.optim as optim  # بهینه‌ساز
from torch.utils.data import Dataset, DataLoader  # دیتاست و لودر
from sklearn.metrics import classification_report, confusion_matrix  # گزارش و ماتریس خطا

# ------------------------------ ثابت‌ها و تنظیمات ------------------------------ #
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]  # نام کلاس‌ها
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}  # نگاشت اسم به عدد

# ------------------------------ دیتاست ساده از پوشه‌ها ------------------------------ #
class FolderMRI(Dataset):  # کلاس دیتاست
    def __init__(self, root_dir, img_size=128):  # سازنده
        self.root_dir = root_dir  # مسیر اصلی
        self.img_size = img_size  # اندازه تصویر
        self.items = []  # لیست نمونه‌ها

        for cname in CLASSES:  # برای هر کلاس
            cdir = os.path.join(root_dir, cname)  # مسیر فولدر کلاس
            if not os.path.isdir(cdir):  # اگر وجود نداشت
                continue  # رد شو
            for fn in os.listdir(cdir):  # فایل‌های داخل فولدر
                if fn.lower().endswith((".png", ".jpg", ".jpeg")):  # اگر تصویر بود
                    self.items.append((os.path.join(cdir, fn), CLASS_TO_ID[cname]))  # ذخیره مسیر و لیبل

    def __len__(self):  # تعداد نمونه‌ها
        return len(self.items)  # طول لیست

    def __getitem__(self, idx):  # گرفتن یک نمونه
        path, y = self.items[idx]  # مسیر و لیبل
        img = Image.open(path).convert("RGB")  # خواندن و تبدیل به رنگی
        img = img.resize((self.img_size, self.img_size))  # تغییر اندازه
        x = np.asarray(img, dtype=np.float32) / 255.0  # تبدیل به عدد اعشاری و نرمال‌سازی
        x = np.transpose(x, (2, 0, 1))  # تبدیل به شکل (کانال، ارتفاع، عرض)
        return torch.tensor(x), torch.tensor(y)  # تبدیل به تنسور

# ------------------------------ یک CNN کوچک و قابل دفاع ------------------------------ #
class SmallCNN(nn.Module):  # مدل ساده
    def __init__(self, num_classes=4):  # سازنده
        super().__init__()  # سازنده پایه
        self.features = nn.Sequential(  # بخش استخراج ویژگی
            nn.Conv2d(3, 16, 3, padding=1),  # کانولوشن ۳×۳
            nn.ReLU(),  # فعال‌ساز
            nn.MaxPool2d(2),  # نصف کردن ابعاد

            nn.Conv2d(16, 32, 3, padding=1),  # کانولوشن ۳×۳
            nn.ReLU(),  # فعال‌ساز
            nn.MaxPool2d(2),  # نصف کردن ابعاد

            nn.Conv2d(32, 64, 3, padding=1),  # کانولوشن ۳×۳
            nn.ReLU(),  # فعال‌ساز
            nn.MaxPool2d(2),  # نصف کردن ابعاد
        )
        self.classifier = nn.Sequential(  # بخش طبقه‌بندی
            nn.Flatten(),  # تخت کردن
            nn.Linear(64 * 16 * 16, 128),  # لایه تمام‌متصل (برای img_size=128)
            nn.ReLU(),  # فعال‌ساز
            nn.Dropout(0.2),  # جلوگیری از بیش‌برازش
            nn.Linear(128, num_classes),  # خروجی کلاس‌ها
        )

    def forward(self, x):  # عبور رو به جلو
        x = self.features(x)  # استخراج ویژگی
        x = self.classifier(x)  # طبقه‌بندی
        return x  # خروجی نهایی

# ------------------------------ محاسبه دقت ------------------------------ #
@torch.no_grad()  # بدون گرادیان
def eval_model(model, loader, device):  # ارزیابی مدل
    model.eval()  # حالت ارزیابی
    all_pred = []  # لیست پیش‌بینی‌ها
    all_true = []  # لیست حقیقت‌ها
    correct = 0  # شمارنده درست‌ها
    total = 0  # کل نمونه‌ها

    for x, y in loader:  # پیمایش دسته‌ها
        x = x.to(device)  # انتقال ورودی به دستگاه
        y = y.to(device)  # انتقال لیبل به دستگاه
        logits = model(x)  # محاسبه خروجی خام
        pred = torch.argmax(logits, dim=1)  # انتخاب بیشترین کلاس
        correct += (pred == y).sum().item()  # جمع درست‌ها
        total += y.numel()  # جمع تعداد
        all_pred.extend(pred.cpu().numpy().tolist())  # ذخیره پیش‌بینی
        all_true.extend(y.cpu().numpy().tolist())  # ذخیره حقیقت

    acc = correct / max(1, total)  # محاسبه دقت
    return acc, np.array(all_true), np.array(all_pred)  # خروجی

# ------------------------------ برنامه اصلی آموزش ------------------------------ #
def main():  # تابع اصلی
    random.seed(1)  # ثابت کردن رندوم
    np.random.seed(1)  # ثابت کردن رندوم
    torch.manual_seed(1)  # ثابت کردن رندوم

    train_dir = "../data/Training"  # مسیر آموزش
    test_dir = "../data/Testing"  # مسیر تست

    img_size = 128  # اندازه ورودی
    batch = 32  # اندازه دسته
    epochs = 10  # تعداد دوره
    lr = 1e-3  # نرخ یادگیری

    device = "cuda" if torch.cuda.is_available() else "cpu"  # انتخاب دستگاه
    print("device:", device)  # چاپ دستگاه

    ds_tr = FolderMRI(train_dir, img_size=img_size)  # ساخت دیتاست آموزش
    ds_ts = FolderMRI(test_dir, img_size=img_size)  # ساخت دیتاست تست
    print("train:", len(ds_tr), "test:", len(ds_ts))  # چاپ تعداد داده

    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, num_workers=0)  # لودر آموزش
    dl_ts = DataLoader(ds_ts, batch_size=batch, shuffle=False, num_workers=0)  # لودر تست

    model = SmallCNN(num_classes=len(CLASSES)).to(device)  # ساخت مدل
    crit = nn.CrossEntropyLoss()  # تابع خطا
    opt = optim.Adam(model.parameters(), lr=lr)  # بهینه‌ساز

    best_acc = 0.0  # بهترین دقت
    os.makedirs("../native", exist_ok=True)  # ساخت فولدر ذخیره

    for ep in range(1, epochs + 1):  # حلقه دوره‌ها
        model.train()  # حالت آموزش
        t0 = time.time()  # زمان شروع
        loss_sum = 0.0  # جمع خطا
        seen = 0  # شمارنده نمونه دیده‌شده

        for x, y in tqdm(dl_tr, desc=f"epoch {ep}/{epochs}"):  # پیمایش دسته‌های آموزش
            x = x.to(device)  # انتقال ورودی
            y = y.to(device)  # انتقال لیبل

            opt.zero_grad()  # صفر کردن گرادیان‌ها
            logits = model(x)  # خروجی خام
            loss = crit(logits, y)  # محاسبه خطا
            loss.backward()  # پس‌انتشار
            opt.step()  # آپدیت وزن‌ها

            loss_sum += float(loss.item()) * x.size(0)  # جمع خطا
            seen += x.size(0)  # جمع نمونه‌ها

        tr_loss = loss_sum / max(1, seen)  # میانگین خطا
        ts_acc, y_true, y_pred = eval_model(model, dl_ts, device)  # ارزیابی روی تست
        dt = time.time() - t0  # زمان دوره

        print(f"ep={ep} loss={tr_loss:.4f} test_acc={ts_acc:.4f} time={dt:.1f}s")  # چاپ خلاصه

        if ts_acc > best_acc:  # اگر بهتر شد
            best_acc = ts_acc  # ذخیره بهترین
            torch.save(model.state_dict(), "../native/mri_cnn_best.pt")  # ذخیره وزن‌ها
            print("saved:", "../native/mri_cnn_best.pt")  # چاپ مسیر

    print("\nbest test acc:", best_acc)  # چاپ بهترین دقت
    print("\nreport:")  # عنوان گزارش
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))  # گزارش کلاس‌ها
    print("confusion matrix:")  # عنوان ماتریس خطا
    print(confusion_matrix(y_true, y_pred))  # چاپ ماتریس خطا

if __name__ == "__main__":  # نقطه ورود
    main()  # اجرای برنامه
