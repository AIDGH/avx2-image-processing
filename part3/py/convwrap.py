import ctypes                                                     # برای صدا زدن کتابخانه‌ی C
import numpy as np                                                # برای آرایه‌ها
import os                                                         # برای مسیر

_here = os.path.dirname(__file__)                                 # پوشه همین فایل
_lib_path = os.path.join(_here, "..", "native", "libconv.so")      # مسیر فایل سویی
lib = ctypes.CDLL(_lib_path)                                       # بارگذاری کتابخانه

lib.conv3x3_f32_1ch_avx2.argtypes = [                              # تعریف نوع آرگومان‌ها
    ctypes.POINTER(ctypes.c_float),                                # ورودی float*
    ctypes.POINTER(ctypes.c_float),                                # خروجی float*
    ctypes.c_int,                                                  # w
    ctypes.c_int,                                                  # h
    ctypes.POINTER(ctypes.c_float),                                # کرنل ۹تایی
]                                                                  # پایان argtypes
lib.conv3x3_f32_1ch_avx2.restype = None                             # خروجی ندارد

def conv3x3_avx2(img: np.ndarray, kernel3x3: np.ndarray) -> np.ndarray:
    img = np.ascontiguousarray(img.astype(np.float32))             # تبدیل به float32 پیوسته
    k = np.ascontiguousarray(kernel3x3.astype(np.float32)).reshape(9) # کرنل ۹تایی پیوسته
    h, w = img.shape                                               # گرفتن اندازه‌ها
    out = np.zeros((h, w), dtype=np.float32)                        # خروجی

    lib.conv3x3_f32_1ch_avx2(                                       # صدا زدن تابع اسمبلی
        img.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),         # اشاره‌گر ورودی
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),         # اشاره‌گر خروجی
        w,                                                          # عرض
        h,                                                          # ارتفاع
        k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),           # اشاره‌گر کرنل
    )                                                               # پایان فراخوانی

    return out                                                      # برگرداندن خروجی
