#include <stdio.h>       // I/O استاندارد (برای printf و ...)
#include <stdlib.h>      // malloc/free و aligned_alloc
#include <string.h>      // strcmp و توابع رشته‌ای
#include <immintrin.h>   // (فعلاً ممکنه لازم نباشه) ولی برای intrinsics/SIMD
#include "conv_utils.h"  // هدر توابع کمکی پروژه

unsigned char clamp_u8(int x) {              // محدود کردن مقدار به بازه‌ی ۰..۲۵۵ برای تبدیل به uint8
    if (x < 0) return 0;                     // هرچی منفی شد -> ۰
    if (x > 255) return 255;                 // هرچی بالای ۲۵۵ شد -> ۲۵۵
    return (unsigned char)x;                 // مقدار مجاز -> تبدیل به unsigned char
}

// RGB interleaved u8 -> R,G,B float جدا
void aos_u8_to_soa_f32(
    const unsigned char* rgb,               // ورودی: آرایه interleaved به فرم RGBRGB...
    float* r, float* g, float* b,           // خروجی: سه آرایه جداگانه برای کانال‌ها (SoA)
    int n_pixels                            // تعداد پیکسل‌ها (w*h)
) {
    for (int i = 0; i < n_pixels; i++) {    // پیمایش همه پیکسل‌ها
        r[i] = (float)rgb[i*3 + 0];         // R پیکسل i -> float
        g[i] = (float)rgb[i*3 + 1];         // G پیکسل i -> float
        b[i] = (float)rgb[i*3 + 2];         // B پیکسل i -> float
    }
}

// R,G,B float جدا -> RGB interleaved u8
void soa_f32_to_aos_u8(
    const float* r, const float* g, const float* b,   // ورودی: کانال‌های جداگانه float
    unsigned char* rgb_out,                            // خروجی: interleaved به فرم RGBRGB...
    int n_pixels                                       // تعداد پیکسل‌ها
) {
    for (int i = 0; i < n_pixels; i++) {              // پیمایش همه پیکسل‌ها
        rgb_out[i*3 + 0] = clamp_u8((int)(r[i] + 0.5f)); // float -> int با گردکردن -> clamp -> u8 (کانال R)
        rgb_out[i*3 + 1] = clamp_u8((int)(g[i] + 0.5f)); // کانال G
        rgb_out[i*3 + 2] = clamp_u8((int)(b[i] + 0.5f)); // کانال B
    }
}

// conv 3x3 روی یک کانال float (نسخه‌ی مرجع C که vectorize نمی‌شود)
__attribute__((optimize("no-tree-vectorize")))        // جلوگیری از auto-vectorize تا baseline واقعاً اسکالر بماند
void conv3x3_f32_1ch(
    const float* in,                                   // ورودی تک‌کاناله (مثلاً R یا G یا B)
    float* out,                                        // خروجی تک‌کاناله
    int w, int h,                                      // عرض و ارتفاع تصویر
    const float k[3][3]                                // کرنل 3x3
) {
    // فقط ناحیه‌ی داخلی را صفر می‌کنیم (مرزها معمولاً بیرون مدیریت می‌شوند)
    for (int y=1; y<h-1; y++)                          // از ردیف 1 تا h-2 (بدون مرز بالا/پایین)
        for(int x=1; x<w-1; x++)                       // از ستون 1 تا w-2 (بدون مرز چپ/راست)
            out[y*w+x] = 0.0f;                         // مقداردهی اولیه برای پیکسل‌های داخلی

    for (int y = 1; y < h - 1; y++) {                  // پیمایش ردیف‌های داخلی
        for (int x = 1; x < w - 1; x++) {              // پیمایش ستون‌های داخلی
            float sum = 0.0f;                          // جمع وزن‌دار همسایه‌های 3x3

            sum += in[(y-1)*w + (x-1)] * k[0][0];       // بالا-چپ
            sum += in[(y-1)*w + (x  )] * k[0][1];       // بالا
            sum += in[(y-1)*w + (x+1)] * k[0][2];       // بالا-راست

            sum += in[(y  )*w + (x-1)] * k[1][0];       // چپ
            sum += in[(y  )*w + (x  )] * k[1][1];       // مرکز
            sum += in[(y  )*w + (x+1)] * k[1][2];       // راست

            sum += in[(y+1)*w + (x-1)] * k[2][0];       // پایین-چپ
            sum += in[(y+1)*w + (x  )] * k[2][1];       // پایین
            sum += in[(y+1)*w + (x+1)] * k[2][2];       // پایین-راست

            out[y*w + x] = sum;                         // ذخیره نتیجه کانولوشن برای پیکسل (x,y)
        }
    }
}

// نسخه‌ی AVX2 (wrapper): بخش برداری در اسمبلی + tail (باقی‌مانده) در C
void conv3x3_f32_1ch_avx2(
    const float* in,                                   // ورودی تک‌کاناله
    float* out,                                        // خروجی تک‌کاناله
    int w, int h,                                      // ابعاد تصویر
    const float k[3][3]                                // کرنل 3x3
){
    conv3x3_f32_1ch_avx2_asm(in, out, w, h, k);         // اجرای بخش برداری داخل فایل اسمبلی (بلوک‌های 8تایی)

    // tail: پیکسل‌های داخلی که در بلوک‌های 8تایی جا نشده‌اند را اسکالر حساب می‌کنیم
    for (int y = 1; y < h-1; y++) {                     // ردیف‌های داخلی
        int x = 1;                                      // شروع از ستون 1 (داخل)
        int last = (w - 1) - 8;                          // آخرین x که asm می‌تواند x..x+7 را کامل حساب کند
        if (last < 1) last = 0;                          // اگر تصویر خیلی باریک بود، عملاً برداری نداریم
        x = (last >= 1) ? (last + 8) : 1;                // اولین x بعد از محدوده‌ی پوشش داده شده توسط asm

        for (; x < w-1; x++) {                           // تا ستون w-2 (داخل)
            float s = 0.0f;                              // جمع وزن‌دار 3x3
            const float* row0 = in + (y-1)*w;            // اشاره‌گر به ردیف بالا
            const float* row1 = in + (y  )*w;            // اشاره‌گر به ردیف وسط
            const float* row2 = in + (y+1)*w;            // اشاره‌گر به ردیف پایین

            s += row0[x-1]*k[0][0] + row0[x]*k[0][1] + row0[x+1]*k[0][2]; // مشارکت ردیف بالا
            s += row1[x-1]*k[1][0] + row1[x]*k[1][1] + row1[x+1]*k[1][2]; // مشارکت ردیف وسط
            s += row2[x-1]*k[2][0] + row2[x]*k[2][1] + row2[x+1]*k[2][2]; // مشارکت ردیف پایین

            out[y*w + x] = s;                             // ذخیره نتیجه tail برای این پیکسل
        }
    }
}
