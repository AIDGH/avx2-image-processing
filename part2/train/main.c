#define _POSIX_C_SOURCE 200809L                                  // فعال‌سازی امکانات زمان پازیکس

#include <stdio.h>                                               // ورودی/خروجی
#include <stdlib.h>                                              // malloc/free/atoi
#include <string.h>                                              // memcpy/strcmp
#include <math.h>                                                // expf/fabsf
#include <stdint.h>                                              // uint32_t
#include <time.h>                                                // clock_gettime
#include "conv_utils.h"                                          // توابع کانولوشن و نسخه اسمبلی

// ------------------------------ تنظیمات کلی ------------------------------ //
#define USE_FEATURES_DEFAULT 1                                   // پیش‌فرض: ویژگی‌ها فعال
#define USE_AVX2_DEFAULT     1                                   // پیش‌فرض: کانولوشن با avx2/اسمبلی
#define CH_DEFAULT           4                                   // تعداد کانال‌های ویژگی (۴ کانال)

// ------------------------------ زمان نانوثانیه ------------------------------ //
static long long now_ns(void) {                                  // گرفتن زمان دقیق برای بنچمارک
    struct timespec ts;                                          // ساختار زمان
    clock_gettime(CLOCK_MONOTONIC, &ts);                         // زمان یکنواخت مناسب بنچمارک
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;     // تبدیل به نانوثانیه
}

// ------------------------------ خواندن عدد 32بیت بیگ‌اندین ------------------------------ //
static uint32_t read_be_u32(FILE* f) {                            // خواندن ۴ بایت به صورت big-endian
    unsigned char b[4];                                          // بافر ۴ بایتی
    if (fread(b, 1, 4, f) != 4) return 0;                        // اگر ناقص خوانده شد خطا
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) |     // ساخت u32 از ۴ بایت
           ((uint32_t)b[2] << 8)  | (uint32_t)b[3];              // ادامه ساخت عدد
}

// ------------------------------ بارگذاری تصاویر MNIST (u8) ------------------------------ //
static unsigned char* load_mnist_images_u8(const char* path, int* n, int* rows, int* cols) { // خواندن فایل تصاویر
    FILE* f = fopen(path, "rb");                                 // باز کردن فایل باینری
    if (!f) { perror("fopen images"); return NULL; }             // اگر باز نشد خطا

    uint32_t magic = read_be_u32(f);                             // کد شناسایی فایل
    uint32_t num   = read_be_u32(f);                             // تعداد تصاویر
    uint32_t r     = read_be_u32(f);                             // تعداد سطر
    uint32_t c     = read_be_u32(f);                             // تعداد ستون

    if (magic != 2051) {                                         // magic استاندارد تصاویر MNIST
        fprintf(stderr, "Bad image magic: %u (expected 2051)\n", magic); // پیام خطا
        fclose(f);                                               // بستن فایل
        return NULL;                                             // خروج
    }

    *n = (int)num;                                               // ثبت تعداد تصاویر
    *rows = (int)r;                                              // ثبت سطرها
    *cols = (int)c;                                              // ثبت ستون‌ها

    size_t total = (size_t)(*n) * (size_t)(*rows) * (size_t)(*cols); // تعداد کل بایت‌ها
    unsigned char* data = (unsigned char*)malloc(total);         // تخصیص حافظه
    if (!data) { fclose(f); return NULL; }                       // اگر تخصیص شکست خورد

    size_t rd = fread(data, 1, total, f);                        // خواندن کل داده
    fclose(f);                                                   // بستن فایل

    if (rd != total) {                                           // بررسی کامل بودن خواندن
        fprintf(stderr, "Image read incomplete: got %zu expected %zu\n", rd, total); // پیام خطا
        free(data);                                              // آزادسازی
        return NULL;                                             // خروج
    }
    return data;                                                 // بازگشت داده تصاویر
}

// ------------------------------ بارگذاری لیبل‌های MNIST (u8) ------------------------------ //
static unsigned char* load_mnist_labels_u8(const char* path, int* n) { // خواندن فایل لیبل‌ها
    FILE* f = fopen(path, "rb");                                 // باز کردن فایل باینری
    if (!f) { perror("fopen labels"); return NULL; }             // اگر باز نشد خطا

    uint32_t magic = read_be_u32(f);                             // کد شناسایی فایل
    uint32_t num   = read_be_u32(f);                             // تعداد لیبل‌ها

    if (magic != 2049) {                                         // magic استاندارد لیبل‌های MNIST
        fprintf(stderr, "Bad label magic: %u (expected 2049)\n", magic); // پیام خطا
        fclose(f);                                               // بستن فایل
        return NULL;                                             // خروج
    }

    *n = (int)num;                                               // ثبت تعداد لیبل‌ها

    unsigned char* lab = (unsigned char*)malloc((size_t)(*n));   // تخصیص برای لیبل‌ها
    if (!lab) { fclose(f); return NULL; }                        // اگر تخصیص شکست خورد

    size_t rd = fread(lab, 1, (size_t)(*n), f);                  // خواندن لیبل‌ها
    fclose(f);                                                   // بستن فایل

    if (rd != (size_t)(*n)) {                                    // بررسی کامل بودن خواندن
        fprintf(stderr, "Label read incomplete: got %zu expected %d\n", rd, *n); // پیام خطا
        free(lab);                                               // آزادسازی
        return NULL;                                             // خروج
    }
    return lab;                                                  // بازگشت لیبل‌ها
}

// ------------------------------ توابع کمکی مدل ------------------------------ //
static int argmax(const float* a, int n) {                        // پیدا کردن اندیس بیشینه
    int im = 0;                                                  // اندیس بیشینه
    float vm = a[0];                                             // مقدار بیشینه
    for (int i = 1; i < n; i++) {                                // پیمایش
        if (a[i] > vm) { vm = a[i]; im = i; }                    // به‌روزرسانی بیشینه
    }
    return im;                                                   // بازگشت اندیس
}

static void softmax_logits(const float* z, float* p, int C) {     // محاسبه سافت‌مکس پایدار
    float m = z[0];                                              // بیشینه برای پایداری
    for (int i = 1; i < C; i++) if (z[i] > m) m = z[i];          // پیدا کردن بیشینه

    float s = 0.0f;                                              // جمع نمایی‌ها
    for (int i = 0; i < C; i++) {                                // محاسبه exp
        p[i] = expf(z[i] - m);                                   // exp با شیفت
        s += p[i];                                               // جمع
    }
    float inv = (s > 1e-20f) ? (1.0f / s) : 0.0f;                // معکوس جمع
    for (int i = 0; i < C; i++) p[i] *= inv;                     // نرمال‌سازی به احتمال
}

static float dotf(const float* a, const float* b, int n) {        // ضرب داخلی ساده
    float s = 0.0f;                                              // جمع
    for (int i = 0; i < n; i++) s += a[i] * b[i];                // جمع ضرب‌ها
    return s;                                                    // نتیجه
}

static void forward_one(                                         // پیش‌بینی برای یک نمونه
    const float* x, int D,                                       // ورودی و طول
    const float* W, const float* b, int C,                       // وزن‌ها و بایاس‌ها
    float* logits, float* probs                                  // خروجی‌ها
) {
    for (int c = 0; c < C; c++) {                                // برای هر کلاس
        const float* wc = W + c*D;                               // اشاره‌گر وزن‌های کلاس
        logits[c] = dotf(wc, x, D) + b[c];                       // محاسبه نمره (لجیت)
    }
    softmax_logits(logits, probs, C);                            // تبدیل نمره‌ها به احتمال
}

static float compute_accuracy(                                   // محاسبه دقت
    const float* X, const unsigned char* y, int N,               // داده و لیبل‌ها
    const float* W, const float* b, int C, int D                 // مدل
) {
    float* logits = (float*)malloc((size_t)C * sizeof(float));   // بافر لجیت
    float* probs  = (float*)malloc((size_t)C * sizeof(float));   // بافر احتمال
    if (!logits || !probs) { free(logits); free(probs); return 0.0f; } // اگر تخصیص شکست خورد

    int correct = 0;                                             // شمارنده درست‌ها
    for (int i = 0; i < N; i++) {                                // پیمایش نمونه‌ها
        const float* x = X + (size_t)i * (size_t)D;              // اشاره‌گر نمونه i
        forward_one(x, D, W, b, C, logits, probs);               // پیش‌بینی
        int pred = argmax(probs, C);                             // برچسب پیش‌بینی
        if (pred == (int)y[i]) correct++;                        // اگر درست بود افزایش
    }

    free(logits);                                                // آزادسازی
    free(probs);                                                 // آزادسازی
    return (float)correct / (float)N;                            // دقت
}

static void confusion_matrix(                                    // چاپ ماتریس خطا
    const float* X, const unsigned char* y, int N,               // داده و لیبل‌ها
    const float* W, const float* b, int C, int D                 // مدل
) {
    int cm[10][10];                                              // ماتریس ۱۰×۱۰
    for (int i = 0; i < 10; i++)                                 // صفر کردن سطرها
        for (int j = 0; j < 10; j++) cm[i][j] = 0;               // صفر کردن خانه‌ها

    float* logits = (float*)malloc((size_t)C * sizeof(float));   // بافر لجیت
    float* probs  = (float*)malloc((size_t)C * sizeof(float));   // بافر احتمال
    if (!logits || !probs) { free(logits); free(probs); return; } // اگر تخصیص شکست خورد

    for (int i = 0; i < N; i++) {                                // برای هر نمونه
        const float* x = X + (size_t)i * (size_t)D;              // اشاره‌گر ورودی
        forward_one(x, D, W, b, C, logits, probs);               // پیش‌بینی
        int pred = argmax(probs, C);                             // کلاس پیش‌بینی‌شده
        int gt = (int)y[i];                                      // کلاس واقعی
        cm[gt][pred]++;                                          // افزایش خانه مربوط
    }

    printf("\nError matrix (row = real, col = prediction)\n");    // توضیح چاپ ماتریس
    for (int r = 0; r < 10; r++) {                               // چاپ سطرها
        for (int c = 0; c < 10; c++) printf("%5d ", cm[r][c]);   // چاپ خانه‌ها
        printf("\n");                                            // خط جدید
    }

    free(logits);                                                // آزادسازی
    free(probs);                                                 // آزادسازی
}

// ------------------------------ ساخت ویژگی‌ها (۴ کانال) ------------------------------ //

static float clamp01(float v) {                                  // محدود کردن مقدار به بازه ۰ تا ۱
    if (v < 0.0f) return 0.0f;                                    // اگر کمتر از صفر بود صفر کن
    if (v > 1.0f) return 1.0f;                                    // اگر بیشتر از یک بود یک کن
    return v;                                                     // در غیر این صورت خود مقدار
}                                                                 // پایان تابع

static void features_4ch_from_mnist_c(                            // ساخت ویژگی‌ها با نسخه C
    const unsigned char* img_u8,                                  // ورودی تصویر ۲۸×۲۸ به صورت بایت
    float* feat_out                                               // خروجی ۴ کانال (۴×۷۸۴)
) {                                                               // شروع تابع
    const int w = 28;                                             // عرض ثابت MNIST
    const int h = 28;                                             // ارتفاع ثابت MNIST
    const int D0 = w * h;                                         // تعداد پیکسل‌ها (۷۸۴)

    static const float kx[3][3] = {                               // کرنل سوبل در راستای x
        { -1, 0, 1 },                                             // سطر ۱
        { -2, 0, 2 },                                             // سطر ۲
        { -1, 0, 1 }                                              // سطر ۳
    };                                                            // پایان کرنل

    static const float ky[3][3] = {                               // کرنل سوبل در راستای y
        { -1, -2, -1 },                                           // سطر ۱
        {  0,  0,  0 },                                           // سطر ۲
        {  1,  2,  1 }                                            // سطر ۳
    };                                                            // پایان کرنل

    static const float klap[3][3] = {                             // کرنل لاپلاسین (جمع ضرایب صفر است)
        { 0,  1, 0 },                                             // سطر ۱
        { 1, -4, 1 },                                             // سطر ۲
        { 0,  1, 0 }                                              // سطر ۳
    };                                                            // پایان کرنل

    float img[28*28];                                             // نسخه float از تصویر
    float gx[28*28];                                              // خروجی گرادیان x
    float gy[28*28];                                              // خروجی گرادیان y
    float lap[28*28];                                             // خروجی لاپلاسین

    for (int i = 0; i < D0; i++) {                                // تبدیل بایت به float
        img[i] = (float)img_u8[i] * (1.0f / 255.0f);              // نرمال‌سازی به ۰..۱
        gx[i] = 0.0f;                                             // صفر اولیه gx
        gy[i] = 0.0f;                                             // صفر اولیه gy
        lap[i] = 0.0f;                                            // صفر اولیه lap
    }                                                             // پایان حلقه

    conv3x3_f32_1ch(img, gx,  w, h, kx);                          // کانولوشن سوبل x
    conv3x3_f32_1ch(img, gy,  w, h, ky);                          // کانولوشن سوبل y
    conv3x3_f32_1ch(img, lap, w, h, klap);                        // کانولوشن لاپلاسین

    const float s_g = 0.125f;                                     // ضریب مقیاس برای گرادیان‌ها (تقریباً /8)
    const float s_l = 0.125f;                                     // ضریب مقیاس برای لاپلاسین (تقریباً /8)

    for (int i = 0; i < D0; i++) feat_out[0*D0 + i] = img[i];     // کانال ۰: خود تصویر (۰..۱)

    for (int i = 0; i < D0; i++) {                                // ساخت کانال ۱
        float v = fabsf(gx[i]) * s_g;                             // قدرمطلق و مقیاس‌کردن gx
        feat_out[1*D0 + i] = clamp01(v);                          // کلمپ به ۰..۱
    }                                                             // پایان کانال ۱

    for (int i = 0; i < D0; i++) {                                // ساخت کانال ۲
        float v = fabsf(gy[i]) * s_g;                             // قدرمطلق و مقیاس‌کردن gy
        feat_out[2*D0 + i] = clamp01(v);                          // کلمپ به ۰..۱
    }                                                             // پایان کانال ۲

    for (int i = 0; i < D0; i++) {                                // ساخت کانال ۳
        float v = fabsf(lap[i]) * s_l;                            // قدرمطلق و مقیاس‌کردن lap
        feat_out[3*D0 + i] = clamp01(v);                          // کلمپ به ۰..۱
    }                                                             // پایان کانال ۳
}                                                                 // پایان تابع

static void features_4ch_from_mnist_avx2(                         // ساخت ویژگی‌ها با نسخه AVX2/اسمبلی
    const unsigned char* img_u8,                                  // ورودی تصویر ۲۸×۲۸ به صورت بایت
    float* feat_out                                               // خروجی ۴ کانال (۴×۷۸۴)
) {                                                               // شروع تابع
    const int w = 28;                                             // عرض ثابت MNIST
    const int h = 28;                                             // ارتفاع ثابت MNIST
    const int D0 = w * h;                                         // تعداد پیکسل‌ها (۷۸۴)

    static const float kx[3][3] = {                               // کرنل سوبل x
        { -1, 0, 1 },                                             // سطر ۱
        { -2, 0, 2 },                                             // سطر ۲
        { -1, 0, 1 }                                              // سطر ۳
    };                                                            // پایان کرنل

    static const float ky[3][3] = {                               // کرنل سوبل y
        { -1, -2, -1 },                                           // سطر ۱
        {  0,  0,  0 },                                           // سطر ۲
        {  1,  2,  1 }                                            // سطر ۳
    };                                                            // پایان کرنل

    static const float klap[3][3] = {                             // کرنل لاپلاسین
        { 0,  1, 0 },                                             // سطر ۱
        { 1, -4, 1 },                                             // سطر ۲
        { 0,  1, 0 }                                              // سطر ۳
    };                                                            // پایان کرنل

    float img[28*28];                                             // نسخه float از تصویر
    float gx[28*28];                                              // خروجی gx
    float gy[28*28];                                              // خروجی gy
    float lap[28*28];                                             // خروجی lap

    for (int i = 0; i < D0; i++) {                                // تبدیل و صفر کردن
        img[i] = (float)img_u8[i] * (1.0f / 255.0f);              // نرمال‌سازی تصویر
        gx[i] = 0.0f;                                             // صفر اولیه gx
        gy[i] = 0.0f;                                             // صفر اولیه gy
        lap[i] = 0.0f;                                            // صفر اولیه lap
    }                                                             // پایان حلقه

    conv3x3_f32_1ch_avx2(img, gx,  w, h, kx);                     // کانولوشن سوبل x با AVX2
    conv3x3_f32_1ch_avx2(img, gy,  w, h, ky);                     // کانولوشن سوبل y با AVX2
    conv3x3_f32_1ch_avx2(img, lap, w, h, klap);                   // کانولوشن لاپلاسین با AVX2

    const float s_g = 0.125f;                                     // ضریب مقیاس گرادیان‌ها
    const float s_l = 0.125f;                                     // ضریب مقیاس لاپلاسین

    for (int i = 0; i < D0; i++) feat_out[0*D0 + i] = img[i];     // کانال ۰: تصویر خام

    for (int i = 0; i < D0; i++) {                                // کانال ۱: gx
        float v = fabsf(gx[i]) * s_g;                             // قدرمطلق و مقیاس gx
        feat_out[1*D0 + i] = clamp01(v);                          // کلمپ به ۰..۱
    }                                                             // پایان کانال ۱

    for (int i = 0; i < D0; i++) {                                // کانال ۲: gy
        float v = fabsf(gy[i]) * s_g;                             // قدرمطلق و مقیاس gy
        feat_out[2*D0 + i] = clamp01(v);                          // کلمپ به ۰..۱
    }                                                             // پایان کانال ۲

    for (int i = 0; i < D0; i++) {                                // کانال ۳: lap
        float v = fabsf(lap[i]) * s_l;                            // قدرمطلق و مقیاس lap
        feat_out[3*D0 + i] = clamp01(v);                          // کلمپ به ۰..۱
    }                                                             // پایان کانال ۳
}                                                                 // پایان تابع

// ------------------------------ بنچمارک ویژگی‌ها ------------------------------ //
static void bench_features(const unsigned char* imgs_u8, int N) { // سنجش سرعت ساخت ویژگی
    float feat[4*28*28];                                         // بافر ویژگی
    int warm = 50;                                               // تعداد گرم‌کردن
    int iters = 2000;                                            // تعداد تکرار واقعی (زیاد تا دقیق شود)

    for (int i = 0; i < warm; i++) {                             // گرم‌کردن نسخه C
        const unsigned char* p = imgs_u8 + (size_t)(i % N) * 784; // انتخاب تصویر
        features_4ch_from_mnist_c(p, feat);                      // اجرای C
    }

    long long t0 = now_ns();                                     // شروع زمان C
    for (int i = 0; i < iters; i++) {                            // تکرار C
        const unsigned char* p = imgs_u8 + (size_t)(i % N) * 784; // انتخاب تصویر
        features_4ch_from_mnist_c(p, feat);                      // اجرای C
    }
    long long t1 = now_ns();                                     // پایان زمان C

    for (int i = 0; i < warm; i++) {                             // گرم‌کردن نسخه avx2
        const unsigned char* p = imgs_u8 + (size_t)(i % N) * 784; // انتخاب تصویر
        features_4ch_from_mnist_avx2(p, feat);                   // اجرای avx2
    }

    long long t2 = now_ns();                                     // شروع زمان avx2
    for (int i = 0; i < iters; i++) {                            // تکرار avx2
        const unsigned char* p = imgs_u8 + (size_t)(i % N) * 784; // انتخاب تصویر
        features_4ch_from_mnist_avx2(p, feat);                   // اجرای avx2
    }
    long long t3 = now_ns();                                     // پایان زمان avx2

    double c_ms   = (t1 - t0) / 1e6;                             // زمان C بر حسب میلی‌ثانیه
    double a_ms   = (t3 - t2) / 1e6;                             // زمان avx2 بر حسب میلی‌ثانیه

    printf("\nbenchmark of similarities(%d):\n", iters);          // عنوان خروجی
    printf("c version: %.3f ms\n", c_ms);                        // چاپ زمان C
    printf("avx2 version: %.3f ms\n", a_ms);                     // چاپ زمان avx2
    printf("acceleration: %.2fx\n", (a_ms > 1e-12) ? (c_ms / a_ms) : 0.0); // چاپ ضریب شتاب
}

// ------------------------------ نمایش چند نمونه از پیش‌بینی ------------------------------ //
static void show_one_per_class(                                         // چاپ یک نمونه از هر کلاس
    const float* X, const unsigned char* y, int N,                       // داده و لیبل‌ها
    const float* W, const float* b, int C, int D                          // مدل
) {                                                                       // شروع تابع
    int seen[10];                                                        // آرایه برای اینکه هر رقم یکبار دیده شود
    for (int i = 0; i < 10; i++) seen[i] = 0;                             // صفر کردن دیده‌شدن‌ها

    float* logits = (float*)malloc((size_t)C * sizeof(float));            // بافر امتیاز خام
    float* probs  = (float*)malloc((size_t)C * sizeof(float));            // بافر احتمال‌ها
    if (!logits || !probs) { free(logits); free(probs); return; }         // اگر حافظه نداد خروج

    printf("\nExample of each number (0-9):\n");                              // عنوان چاپ

    int done = 0;                                                        // شمارنده تعداد کلاس‌های تکمیل‌شده
    for (int i = 0; i < N && done < 10; i++) {                            // پیمایش دیتاست تا وقتی ۱۰ رقم پیدا شود
        int gt = (int)y[i];                                               // لیبل واقعی نمونه
        if (gt < 0 || gt > 9) continue;                                   // اگر خارج از بازه بود رد کن
        if (seen[gt]) continue;                                           // اگر این رقم قبلاً چاپ شده رد کن

        const float* x = X + (size_t)i * (size_t)D;                       // اشاره‌گر به ورودی
        forward_one(x, D, W, b, C, logits, probs);                        // اجرای مدل
        int pred = argmax(probs, C);                                       // انتخاب کلاس پیش‌بینی

        printf("sample %d : %d -> %d   (p = %.3f)\n", i, (int)y[i], pred, probs[pred]); // چاپ نتیجه ترین نمونه‌ای از هر کلاس

        seen[gt] = 1;                                                     // این رقم را دیده‌شده علامت بزن
        done++;                                                           // یکی به شمارنده اضافه کن
    }

    free(probs);                                                          // آزادسازی
    free(logits);                                                         // آزادسازی
}                                                                         // پایان تابع

// ------------------------------ ذخیره مدل روی دیسک ------------------------------ //
static int save_model(const char* path, const float* W, const float* b, int C, int D) { // ذخیره وزن‌ها و بایاس‌ها
    FILE* f = fopen(path, "wb");                                  // باز کردن فایل خروجی باینری
    if (!f) { perror("save_model fopen"); return 0; }             // اگر باز نشد خطا
    fwrite(&C, sizeof(int), 1, f);                                 // نوشتن تعداد کلاس‌ها
    fwrite(&D, sizeof(int), 1, f);                                 // نوشتن طول ورودی
    fwrite(W, sizeof(float), (size_t)C*(size_t)D, f);              // نوشتن کل وزن‌ها
    fwrite(b, sizeof(float), (size_t)C, f);                        // نوشتن بایاس‌ها
    fclose(f);                                                     // بستن فایل
    return 1;                                                      // اعلام موفقیت
}

// ------------------------------ بارگذاری مدل از دیسک ------------------------------ //
static int load_model(const char* path, float** W, float** b, int* C, int* D) { // خواندن مدل ذخیره‌شده
    FILE* f = fopen(path, "rb");                                   // باز کردن فایل ورودی باینری
    if (!f) { perror("load_model fopen"); return 0; }              // اگر باز نشد خطا

    int c=0,d=0;                                                   // متغیرهای موقت برای خواندن
    if (fread(&c, sizeof(int), 1, f) != 1) { fclose(f); return 0; } // خواندن تعداد کلاس‌ها
    if (fread(&d, sizeof(int), 1, f) != 1) { fclose(f); return 0; } // خواندن طول ورودی

    float* w = (float*)malloc((size_t)c*(size_t)d*sizeof(float));   // تخصیص حافظه وزن‌ها
    float* bb = (float*)malloc((size_t)c*sizeof(float));            // تخصیص حافظه بایاس‌ها
    if (!w || !bb) { free(w); free(bb); fclose(f); return 0; }      // اگر تخصیص شکست خورد

    if (fread(w, sizeof(float), (size_t)c*(size_t)d, f) != (size_t)c*(size_t)d) { free(w); free(bb); fclose(f); return 0; } // خواندن وزن‌ها
    if (fread(bb, sizeof(float), (size_t)c, f) != (size_t)c) { free(w); free(bb); fclose(f); return 0; } // خواندن بایاس‌ها

    fclose(f);                                                      // بستن فایل
    *C = c; *D = d;                                                 // برگرداندن ابعاد مدل
    *W = w; *b = bb;                                                // برگرداندن اشاره‌گرها
    return 1;                                                       // اعلام موفقیت
}

// ------------------------------ خواندن یک تصویر خام 28×28 ------------------------------ //
static int load_raw_28x28_u8(const char* path, unsigned char out[28*28]) { // خواندن فایل خام باینری
    FILE* f = fopen(path, "rb");                                   // باز کردن فایل
    if (!f) { perror("open raw"); return 0; }                      // اگر باز نشد خطا
    size_t rd = fread(out, 1, 28*28, f);                            // خواندن دقیق ۷۸۴ بایت
    fclose(f);                                                      // بستن فایل
    if (rd != 28*28) {                                              // بررسی کامل بودن خواندن
        fprintf(stderr, "raw read incomplete: got %zu expected %d\n", rd, 28*28); // پیام خطا
        return 0;                                                   // خروج ناموفق
    }
    return 1;                                                       // خروج موفق
}

// ------------------------------ پیش‌بینی برای یک تصویر 28×28 ------------------------------ //
static int predict_one_u8img(                                       // پیش‌بینی یک نمونه ورودی
    const unsigned char img_u8[28*28],                              // تصویر ورودی (بایت)
    int use_features, int use_avx2,                                 // انتخاب ویژگی و avx2
    const float* W, const float* b, int C                           // وزن‌ها/بایاس‌ها و تعداد کلاس
){
    const int D0 = 28*28;                                           // طول تصویر خام
    const int CH = use_features ? 4 : 1;                            // تعداد کانال‌های ورودی مدل
    const int D  = CH * D0;                                         // طول نهایی بردار ورودی

    float* x = (float*)malloc((size_t)D * sizeof(float));           // ساخت بردار ورودی مدل
    if (!x) return -1;                                              // اگر حافظه نداد خروج

    if (use_features) {                                             // اگر حالت ویژگی فعال باشد
        float feat[4*28*28];                                        // بافر ویژگی موقت
        if (use_avx2) features_4ch_from_mnist_avx2(img_u8, feat);   // ساخت ویژگی با avx2
        else          features_4ch_from_mnist_c(img_u8, feat);      // ساخت ویژگی با C
        memcpy(x, feat, (size_t)D * sizeof(float));                 // کپی به بردار ورودی
    } else {                                                        // اگر ورودی خام باشد
        for (int i = 0; i < D0; i++) x[i] = (float)img_u8[i] * (1.0f/255.0f); // نرمال‌سازی پیکسل‌ها
    }

    float* logits = (float*)malloc((size_t)C * sizeof(float));      // بافر امتیاز خام کلاس‌ها
    float* probs  = (float*)malloc((size_t)C * sizeof(float));      // بافر احتمال کلاس‌ها
    if (!logits || !probs) { free(x); free(logits); free(probs); return -1; } // اگر تخصیص شکست خورد

    forward_one(x, D, W, b, C, logits, probs);                      // اجرای مدل و محاسبه احتمال‌ها
    int pred = argmax(probs, C);                                    // انتخاب کلاس با بیشترین احتمال

    printf("prediction=%d  confidence=%.3f\n", pred, probs[pred]);  // چاپ نتیجه و میزان اطمینان

    free(probs);                                                    // آزادسازی
    free(logits);                                                   // آزادسازی
    free(x);                                                        // آزادسازی
    return pred;                                                    // بازگشت رقم پیش‌بینی‌شده
}

static void shuffle_int(int* a, int n) {           // تابع برای درهم‌ریختن آرایه اندیس‌ها
    for (int i = n - 1; i > 0; i--) {              // از آخر آرایه به سمت ابتدا حرکت می‌کنیم
        int j = rand() % (i + 1);                  // یک اندیس تصادفی بین ۰ تا i انتخاب می‌کنیم
        int t = a[i];                              // مقدار خانه i را موقت ذخیره می‌کنیم
        a[i] = a[j];                               // مقدار خانه j را به خانه i می‌بریم
        a[j] = t;                                  // مقدار موقت را در خانه j قرار می‌دهیم (جابجایی کامل می‌شود)
    }
}
// ------------------------------ main ------------------------------ //
int main(int argc, char** argv) {                                  // نقطه شروع برنامه
    srand(1);                                                                  // ثابت نگه داشتن رندوم برای تکرارپذیری

    const char* train_img_path = "mnist_idx/train-images-idx3-ubyte";  // مسیر تصاویر آموزش
    const char* train_lab_path = "mnist_idx/train-labels-idx1-ubyte";  // مسیر لیبل آموزش
    const char* test_img_path  = "mnist_idx/t10k-images-idx3-ubyte";   // مسیر تصاویر تست
    const char* test_lab_path  = "mnist_idx/t10k-labels-idx1-ubyte";   // مسیر لیبل تست

    int C = 10;                                                    // تعداد کلاس‌ها (۰ تا ۹)
    int epochs = 5;                                                // تعداد دوره آموزش
    int train_limit = 20000;                                       // تعداد نمونه آموزش
    int test_limit  = 10000;                                       // تعداد نمونه تست
    float lr = 0.1f;                                               // نرخ یادگیری
    float l2 = 1e-4f;                                              // منظم‌سازی
    int batch = 50;                                                // اندازه مینی‌بچ

    int use_features = USE_FEATURES_DEFAULT;                       // فعال/غیرفعال کردن ویژگی‌ها
    int use_avx2     = USE_AVX2_DEFAULT;                           // انتخاب نسخه avx2 یا C
    int do_bench     = 1;                                          // اجرای بنچمارک (برای گزارش)

    // ورودی خط فرمان: ./prog [epochs] [train_limit] [lr] [use_features] [use_avx2]
    if (argc >= 2) epochs = atoi(argv[1]);                         // خواندن epochs
    if (argc >= 3) train_limit = atoi(argv[2]);                    // خواندن تعداد train
    if (argc >= 4) lr = (float)atof(argv[3]);                      // خواندن lr
    if (argc >= 5) use_features = atoi(argv[4]);                   // ۰ یا ۱ برای ویژگی
    if (argc >= 6) use_avx2 = atoi(argv[5]);                       // ۰ یا ۱ برای avx2

    // حالت پیش‌بینی:  ./mnist_lr predict model.bin img.raw [use_features] [use_avx2]
    if (argc >= 2 && strcmp(argv[1], "predict") == 0) {            // اگر حالت پیش‌بینی انتخاب شد
        if (argc < 4) {                                            // اگر آرگومان‌ها کم بود
            printf("Usage: %s predict model.bin img.raw [use_features] [use_avx2]\n", argv[0]); // راهنما
            return 1;                                              // خروج با خطا
        }
        const char* model_path = argv[2];                          // مسیر فایل مدل
        const char* img_path   = argv[3];                          // مسیر تصویر خام 28×28
        int uf = (argc >= 5) ? atoi(argv[4]) : USE_FEATURES_DEFAULT; // انتخاب ویژگی‌ها
        int ua = (argc >= 6) ? atoi(argv[5]) : USE_AVX2_DEFAULT;     // انتخاب avx2

        float* W2 = NULL;                                          // اشاره‌گر وزن‌های مدل خوانده‌شده
        float* b2 = NULL;                                          // اشاره‌گر بایاس‌های مدل خوانده‌شده
        int C2=0, D2=0;                                            // ابعاد مدل خوانده‌شده

        if (!load_model(model_path, &W2, &b2, &C2, &D2)) {          // بارگذاری مدل
            fprintf(stderr, "failed to load model\n");             // پیام خطا
            return 1;                                              // خروج
        }

        unsigned char img_u8[28*28];                               // بافر تصویر خام
        if (!load_raw_28x28_u8(img_path, img_u8)) {                // خواندن تصویر خام
            free(W2); free(b2);                                    // آزادسازی مدل
            return 1;                                              // خروج
        }

        predict_one_u8img(img_u8, uf, ua, W2, b2, C2);             // انجام پیش‌بینی و چاپ نتیجه

        free(W2); free(b2);                                        // آزادسازی مدل
        return 0;                                                  // پایان حالت پیش‌بینی
    }

    int ntr, rows, cols;                                           // مشخصات train
    unsigned char* tr_u8 = load_mnist_images_u8(train_img_path, &ntr, &rows, &cols); // خواندن train images
    if (!tr_u8) return 1;                                          // اگر خطا شد خروج

    int ntr_lab;                                                   // تعداد لیبل train
    unsigned char* tr_y = load_mnist_labels_u8(train_lab_path, &ntr_lab); // خواندن لیبل train
    if (!tr_y) { free(tr_u8); return 1; }                          // اگر خطا شد خروج

    int nts, rows2, cols2;                                         // مشخصات test
    unsigned char* ts_u8 = load_mnist_images_u8(test_img_path, &nts, &rows2, &cols2); // خواندن test images
    if (!ts_u8) { free(tr_u8); free(tr_y); return 1; }             // اگر خطا شد خروج

    int nts_lab;                                                   // تعداد لیبل test
    unsigned char* ts_y = load_mnist_labels_u8(test_lab_path, &nts_lab); // خواندن لیبل test
    if (!ts_y) { free(tr_u8); free(tr_y); free(ts_u8); return 1; } // اگر خطا شد خروج

    if (ntr != ntr_lab || nts != nts_lab || rows != 28 || cols != 28 || rows2 != 28 || cols2 != 28) { // بررسی سازگاری
        fprintf(stderr, "Dataset mismatch: train (%d/%d) test (%d/%d) dims (%dx%d)\n", ntr, ntr_lab, nts, nts_lab, rows, cols); // پیام خطا
        free(tr_u8); free(tr_y); free(ts_u8); free(ts_y);          // آزادسازی
        return 1;                                                  // خروج
    }

    if (train_limit > ntr) train_limit = ntr;                      // محدود کردن train_limit
    if (test_limit > nts) test_limit = nts;                        // محدود کردن test_limit

    int D0 = rows * cols;                                          // طول ورودی خام (784)
    int CH = use_features ? CH_DEFAULT : 1;                        // اگر ویژگی فعال باشد ۴ کانال
    int D  = CH * D0;                                              // طول نهایی ورودی به مدل

    if (do_bench && use_features) {                                // اگر ویژگی فعال است بنچمارک بگیر
        bench_features(tr_u8, train_limit);                        // بنچمارک روی train
    }

    float* Xtr = (float*)malloc((size_t)train_limit * (size_t)D * sizeof(float)); // داده train
    float* Xts = (float*)malloc((size_t)test_limit  * (size_t)D * sizeof(float)); // داده test
    if (!Xtr || !Xts) {                                            // اگر تخصیص شکست خورد
        fprintf(stderr, "alloc X failed\n");                       // پیام خطا
        free(tr_u8); free(tr_y); free(ts_u8); free(ts_y);          // آزادسازی
        free(Xtr); free(Xts);                                      // آزادسازی
        return 1;                                                  // خروج
    }

    // پر کردن Xtr  // بخش مهم
    if (use_features) {                                            // اگر ویژگی فعال است
        float feat[4*28*28];                                       // بافر ویژگی ۴ کانال
        for (int i = 0; i < train_limit; i++) {                    // پیمایش تصاویر train
            const unsigned char* src = tr_u8 + (size_t)i * (size_t)D0; // اشاره‌گر تصویر i
            float* dst = Xtr + (size_t)i * (size_t)D;              // اشاره‌گر خروجی ویژگی‌ها
            if (use_avx2) features_4ch_from_mnist_avx2(src, feat); // ساخت ویژگی با avx2
            else          features_4ch_from_mnist_c(src, feat);    // ساخت ویژگی با C
            memcpy(dst, feat, (size_t)D * sizeof(float));          // کپی به Xtr
            for (int j = 0; j < D; j++) {                                             // پیمایش کل ویژگی‌ها
                if (!isfinite(dst[j])) {                                              // اگر مقدار نرمال نیست (NaN/inf)
                    printf("bad feature at sample=%d j=%d val=%f\n", i, j, dst[j]);    // چاپ خطا
                    exit(1);                                                          // خروج فوری
                }
            }
        }
    } else {                                                       // اگر ویژگی خام باشد
        for (int i = 0; i < train_limit; i++) {                    // پیمایش تصاویر train
            const unsigned char* src = tr_u8 + (size_t)i * (size_t)D0; // اشاره‌گر تصویر i
            float* dst = Xtr + (size_t)i * (size_t)D;              // اشاره‌گر خروجی
            for (int j = 0; j < D0; j++) dst[j] = (float)src[j] * (1.0f / 255.0f); // نرمال‌سازی خام
        }
    }

    // پر کردن Xts  // بخش مهم
    if (use_features) {                                            // اگر ویژگی فعال است
        float feat[4*28*28];                                       // بافر ویژگی ۴ کانال
        for (int i = 0; i < test_limit; i++) {                     // پیمایش تصاویر test
            const unsigned char* src = ts_u8 + (size_t)i * (size_t)D0; // اشاره‌گر تصویر i
            float* dst = Xts + (size_t)i * (size_t)D;              // اشاره‌گر خروجی ویژگی‌ها
            if (use_avx2) features_4ch_from_mnist_avx2(src, feat); // ساخت ویژگی با avx2
            else          features_4ch_from_mnist_c(src, feat);    // ساخت ویژگی با C
            memcpy(dst, feat, (size_t)D * sizeof(float));          // کپی به Xts
        }
    } else {                                                       // اگر ویژگی خام باشد
        for (int i = 0; i < test_limit; i++) {                     // پیمایش تصاویر test
            const unsigned char* src = ts_u8 + (size_t)i * (size_t)D0; // اشاره‌گر تصویر i
            float* dst = Xts + (size_t)i * (size_t)D;              // اشاره‌گر خروجی
            for (int j = 0; j < D0; j++) dst[j] = (float)src[j] * (1.0f / 255.0f); // نرمال‌سازی خام
        }
    }

    int* perm = (int*)malloc((size_t)train_limit * sizeof(int));   // تخصیص حافظه برای آرایه اندیس‌ها

    if (!perm) {                                                   // اگر حافظه تخصیص داده نشد
        fprintf(stderr, "alloc perm failed\n");                     // چاپ پیام خطا
        return 1;                                                   // خروج از برنامه
    }

    for (int i = 0; i < train_limit; i++)                           // پر کردن آرایه اندیس‌ها
        perm[i] = i;                                                // هر خانه برابر با شماره خودش قرار می‌گیرد

    srand(1);                                                       // مقداردهی اولیه مولد اعداد تصادفی (ثابت برای تکرارپذیری)

    float* W = (float*)malloc((size_t)C * (size_t)D * sizeof(float)); // وزن‌ها
    float* b = (float*)malloc((size_t)C * sizeof(float));             // بایاس‌ها
    if (!W || !b) {                                                // اگر تخصیص شکست خورد
        fprintf(stderr, "alloc model failed\n");                   // پیام خطا
        free(tr_u8); free(tr_y); free(ts_u8); free(ts_y);          // آزادسازی
        free(Xtr); free(Xts);                                      // آزادسازی
        free(W); free(b);                                          // آزادسازی
        return 1;                                                  // خروج
    }

    // for (int i = 0; i < C*D; i++) {                                           // پیمایش کل وزن‌ها
    //     float r = (float)rand() / (float)RAND_MAX;                            // عدد رندوم ۰ تا ۱
    //     W[i] = (r - 0.5f) * 0.01f;                                            // تبدیل به بازه حدوداً -۰.۰۰۵ تا +۰.۰۰۵
    // }                                                                          // پایان مقداردهی وزن‌ها
    for (int i = 0; i < C*D; i++)   W[i] = 0.0f;                     // مقداردهی اولیه وزن‌ها
    for (int i = 0; i < C; i++)   b[i] = 0.0f;                     // مقداردهی اولیه بایاس‌ها

    float* logits = (float*)malloc((size_t)C * sizeof(float));     // بافر لجیت
    float* probs  = (float*)malloc((size_t)C * sizeof(float));     // بافر احتمال
    float* dW     = (float*)malloc((size_t)C * (size_t)D * sizeof(float)); // گرادیان وزن
    float* db     = (float*)malloc((size_t)C * sizeof(float));     // گرادیان بایاس
    if (!logits || !probs || !dW || !db) {                         // اگر تخصیص شکست خورد
        fprintf(stderr, "alloc buffers failed\n");                 // پیام خطا
        free(tr_u8); free(tr_y); free(ts_u8); free(ts_y);          // آزادسازی
        free(Xtr); free(Xts);                                      // آزادسازی
        free(W); free(b);                                          // آزادسازی
        free(logits); free(probs); free(dW); free(db);             // آزادسازی
        return 1;                                                  // خروج
    }

    printf("MNIST: train=%d test=%d D=%d epochs=%d batch=%d lr=%.3f l2=%.1e features=%d avx2=%d\n",
           train_limit, test_limit, D, epochs, batch, lr, l2, use_features, use_avx2); // چاپ تنظیمات

    // ------------------------------ آموزش مدل ------------------------------ //
    for (int ep = 1; ep <= epochs; ep++) {                         // حلقه دوره‌ها

        shuffle_int(perm, train_limit);   // درهم‌ریختن ترتیب نمونه‌های آموزش در ابتدای هر دوره

        int nb = (train_limit + batch - 1) / batch;                // تعداد مینی‌بچ‌ها

        for (int bi = 0; bi < nb; bi++) {                          // پیمایش مینی‌بچ‌ها
            int i0 = bi * batch;                                   // شروع اندیس
            int i1 = i0 + batch;                                   // پایان اندیس
            if (i1 > train_limit) i1 = train_limit;                // اصلاح پایان
            int bs = i1 - i0;                                      // اندازه واقعی مینی‌بچ

            for (int i = 0; i < C*D; i++) dW[i] = 0.0f;            // صفر کردن گرادیان وزن
            for (int i = 0; i < C;   i++) db[i] = 0.0f;            // صفر کردن گرادیان بایاس

            for (int i = i0; i < i1; i++) {                        // جمع گرادیان‌ها
                int ii = perm[i];                                  // گرفتن اندیس واقعی پس از شافل
                const float* x = Xtr + (size_t)ii * (size_t)D;     // گرفتن نمونه آموزش با ترتیب جدید
                int y = (int)tr_y[ii];                             // گرفتن لیبل متناظر با همان اندیس

                forward_one(x, D, W, b, C, logits, probs);         // پیش‌بینی احتمال‌ها

                for (int c = 0; c < C; c++) {                      // برای هر کلاس
                    float g = probs[c] - (float)(c == y);          // گرادیان لجیت: p - onehot
                    db[c] += g;                                    // جمع گرادیان بایاس
                    float* dWc = dW + (size_t)c * (size_t)D;       // اشاره‌گر گرادیان وزن کلاس
                    for (int j = 0; j < D; j++) dWc[j] += g * x[j]; // جمع گرادیان وزن‌ها
                }
            }

            float invbs = 1.0f / (float)bs;                        // معکوس اندازه مینی‌بچ
            for (int c = 0; c < C; c++) {                          // آپدیت برای هر کلاس
                float* Wc  = W  + (size_t)c * (size_t)D;           // وزن‌های کلاس
                float* dWc = dW + (size_t)c * (size_t)D;           // گرادیان وزن کلاس
                for (int j = 0; j < D; j++) {                      // آپدیت وزن‌ها
                    float grad = dWc[j] * invbs + l2 * Wc[j];      // گرادیان با L2
                    Wc[j] -= lr * grad;                            // کم کردن در جهت گرادیان
                }
                b[c] -= lr * (db[c] * invbs);                      // آپدیت بایاس
            }
        }

        float acc_tr = compute_accuracy(Xtr, tr_y, train_limit, W, b, C, D); // دقت train
        float acc_ts = compute_accuracy(Xts, ts_y, test_limit,  W, b, C, D); // دقت test
        printf("epoch %d/%d  train_acc=%.4f  test_acc=%.4f\n", ep, epochs, acc_tr, acc_ts); // چاپ
    }

    save_model("model.bin", W, b, C, D);
    printf("model saved to model.bin\n");

    // ------------------------------ خروجی‌های تحلیلی ------------------------------ //
    show_one_per_class(Xts, ts_y, test_limit, W, b, C, D);         // چاپ یک نمونه از هر رقم
    confusion_matrix(Xts, ts_y, test_limit, W, b, C, D);           // چاپ ماتریس خطا

    // ------------------------------ پاکسازی ------------------------------ //
    free(perm);                                                   // آزادسازی آرایه اندیس‌های شافل
    free(db);                                                     // آزادسازی
    free(dW);                                                     // آزادسازی
    free(probs);                                                  // آزادسازی
    free(logits);                                                 // آزادسازی
    free(b);                                                      // آزادسازی
    free(W);                                                      // آزادسازی
    free(Xts);                                                    // آزادسازی
    free(Xtr);                                                    // آزادسازی
    free(ts_y);                                                   // آزادسازی
    free(ts_u8);                                                  // آزادسازی
    free(tr_y);                                                   // آزادسازی
    free(tr_u8);                                                  // آزادسازی
    return 0;                                                     // پایان برنامه
}
