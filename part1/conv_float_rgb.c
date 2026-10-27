#define _POSIX_C_SOURCE 200809L                // فعال کردن APIهای POSIX مثل clock_gettime روی بعضی سیستم‌ها

#include <stdio.h>                              // printf / perror
#include <stdlib.h>                             // malloc/free/aligned_alloc
#include <string.h>                             // strcmp
#include <time.h>                               // clock_gettime + timespec
#include "conv_utils.h"                         // توابع تبدیل RGB و conv ها (C/AVX/ASM)

#define USE_AVX2 1                              // سوییچ انتخاب نسخه‌ی AVX2 (فعلاً در کد اصلی کامنت شده)

// Hand Made (IDK what is it tbh :D)
float k[3][3] = {                               // کرنل 3 در 3 که روی هر کانال جدا اعمال میشه
    { 2, 10,  -3 },
    { -1, -6, 3 },
    { 5, -8, -1 }
};
// Sharpen
// float k[3][3] = {
//     {  0, -1,  0 },
//     { -1,  5, -1 },
//     {  0, -1,  0 }
// };
// Blur
// float k[3][3] = {
//     { 1, 1, 1 },
//     { 1, 1, 1 },
//     { 1, 1, 1 }
// };
// Gaussian Blur
// float k[3][3] = {
//     { 1, 2, 1 },
//     { 2, 4, 2 },
//     { 1, 2, 1 }
// };
// Edge Detection X
// float k[3][3] = {
//     { -1, 0, 1 },
//     { -2, 0, 2 },
//     { -1, 0, 1 }
// };
// Edge Detection Y
// float k[3][3] = {
//     { -1, -2, -1 },
//     {  0,  0,  0 },
//     {  1,  2,  1 }
// };
// Laplacian
// float k[3][3] = {
//     {  0,  1,  0 },
//     {  1, -4,  1 },
//     {  0,  1,  0 }
// };
// or
// float k[3][3] = {
//     { -1, -1, -1 },
//     { -1,  8, -1 },
//     { -1, -1, -1 }
// };
// Emboss
// float k[3][3] = {
//     { -2, -1, 0 },
//     { -1,  1, 1 },
//     {  0,  1, 2 }
// };
// Edge Enhance
// float k[3][3] = {
//     { 0,  0,  0 },
//     { -1, 1,  0 },
//     { 0,  0,  0 }
// };
// High Pass
// float k[3][3] = {
//     { -1, -1, -1 },
//     { -1,  9, -1 },
//     { -1, -1, -1 }
// };
// Motion Blur
// float k[3][3] = {
//     { 0, 0, 0 },
//     { 1, 1, 1 },
//     { 0, 0, 0 }
// };

int kernelCount = 0;                             // جمع ضرایب کرنل (برای نرمال‌سازی/کنترل روشنایی)

static inline long long now_ns(void) {           // تایمر دقیق بر حسب نانوثانیه (برای benchmark)
    struct timespec ts;                          // ساختار زمان POSIX
    clock_gettime(CLOCK_MONOTONIC, &ts);         // زمان یکنواخت (برای بنچمارک بهتر از time())
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec; // تبدیل sec + nsec به ns
}

unsigned char* read_ppm_p6(const char* path, int* w, int* h) {  // خواندن فایل PPM نوع P6 (RGB خام)
    FILE* f = fopen(path, "rb");                 // باز کردن فایل به صورت باینری
    if (!f) { perror("fopen"); return NULL; }    // اگر فایل باز نشد، خروج

    char magic[3];                               // برای نگه داشتن "P6"
    if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P6") != 0) { // چک کردن هدر P6
        printf("Not P6\n");                      // فرمت فایل درست نیست
        fclose(f);                               // بستن فایل
        return NULL;                             // شکست
    }

    int c = fgetc(f);                            // خواندن اولین کاراکتر بعد از magic
    while (c == '#') {                           // اگر کامنت PPM وجود داشت
        while (fgetc(f) != '\n');                // تا آخر خط کامنت بخون و دور بریز
        c = fgetc(f);                            // کاراکتر بعدی برای ادامه بررسی
    }
    ungetc(c, f);                                // یک کاراکتر جلو رفتیم؛ برگردونش تا fscanf درست بخونه

    int maxv;                                    // max value (معمولاً 255)
    if (fscanf(f, "%d %d %d", w, h, &maxv) != 3) { // خواندن width/height/maxval
        printf("Bad header\n");                  // هدر خراب یا غیرمنتظره
        fclose(f);                               // بستن فایل
        return NULL;                             // شکست
    }

    fgetc(f);                                    // یک وایت اسپیس بعد از هدر رو رد می‌کنیم تا به دیتا برسیم

    int size = (*w) * (*h) * 3;                  // اندازه داده RGB (سه کانال برای هر پیکسل)
    unsigned char* data = malloc(size);          // تخصیص بافر برای دیتا
    if (!data) {                                 // اگر malloc شکست خورد
        printf("malloc failed\n");               // پیام خطا
        fclose(f);                               // بستن فایل
        return NULL;                             // شکست
    }

    size_t read_bytes = fread(data, 1, size, f); // خواندن کل داده پیکسل‌ها
    if (read_bytes != (size_t)size) {            // اگر فایل ناقص بود
        printf("File read incomplete\n");        // پیام خطا
        free(data);                              // آزاد کردن بافر
        fclose(f);                               // بستن فایل
        return NULL;                             // شکست
    }

    fclose(f);                                   // بستن فایل
    return data;                                 // برگرداندن بافر RGB
}

int write_ppm_p6(const char* path, unsigned char* data, int w, int h) { // نوشتن خروجی به PPM نوع P6
    FILE* f = fopen(path, "wb");                 // باز کردن فایل خروجی به صورت باینری
    if (!f) return 0;                            // اگر باز نشد -> شکست
    fprintf(f, "P6\n%d %d\n255\n", w, h);         // نوشتن هدر استاندارد P6
    fwrite(data, 1, w*h*3, f);                   // نوشتن دیتای RGB خام
    fclose(f);                                   // بستن فایل
    return 1;                                    // موفق
}

// --- Benchmark settings ---
static void zero_borders(float* out, int w, int h) { // صفر کردن فقط مرزها (برای جلوگیری از هزینه‌ی صفر کردن کل تصویر)
    for (int x = 0; x < w; x++) {                // پیمایش ستون‌ها
        out[0*w + x]       = 0.0f;               // سطر اول (بالا)
        out[(h-1)*w + x]   = 0.0f;               // سطر آخر (پایین)
    }
    for (int y = 1; y < h-1; y++) {              // پیمایش سطرهای میانی
        out[y*w + 0]       = 0.0f;               // ستون اول (چپ)
        out[y*w + (w-1)]   = 0.0f;               // ستون آخر (راست)
    }
}

void benchmark (                                   // بنچمارک: اندازه‌گیری زمان نسخه C و نسخه AVX2/ASM
    float* R,                                      // کانال R ورودی (float)
    float* G,                                      // کانال G ورودی
    float* B,                                      // کانال B ورودی
    float* Ro,                                     // خروجی R
    float* Go,                                     // خروجی G
    float* Bo,                                     // خروجی B
    int w,                                         // width
    int h                                          // height
){
    int warmup = 5;                               // چند بار اجرای اولیه برای warmup (cache/boost/...)
    int iters  = 50;                              // اگر تصویر بزرگه (مثلاً 1024) میشه کمتر کرد

    // Warmup C
    zero_borders(Ro,w,h); zero_borders(Go,w,h); zero_borders(Bo,w,h); // آماده‌سازی مرزها (داخل رو conv می‌نویسه)
    for (int i = 0; i < warmup; i++) {             // چند اجرای اولیه برای پایدار شدن زمان
        conv3x3_f32_1ch(R, Ro, w, h, k);            // کانولوشن C روی کانال R
        conv3x3_f32_1ch(G, Go, w, h, k);            // کانولوشن C روی کانال G
        conv3x3_f32_1ch(B, Bo, w, h, k);            // کانولوشن C روی کانال B
    }

    // Measure C
    long long t0 = now_ns();                       // زمان شروع اندازه‌گیری C
    zero_borders(Ro,w,h); zero_borders(Go,w,h); zero_borders(Bo,w,h); // مرزها قبل از اندازه‌گیری
    for (int i = 0; i < iters; i++) {              // تکرار برای افزایش دقت زمان‌گیری
        conv3x3_f32_1ch(R, Ro, w, h, k);            // اجرای C (R)
        conv3x3_f32_1ch(G, Go, w, h, k);            // اجرای C (G)
        conv3x3_f32_1ch(B, Bo, w, h, k);            // اجرای C (B)
    }
    long long t1 = now_ns();                       // زمان پایان اندازه‌گیری C
    double c_ms = (t1 - t0) / 1e6;                 // تبدیل ns به ms

    // Warmup AVX2
    zero_borders(Ro,w,h); zero_borders(Go,w,h); zero_borders(Bo,w,h); // مرزها قبل از warmup AVX2
    for (int i = 0; i < warmup; i++) {             // warmup برای نسخه برداری/اسمبلی
        conv3x3_f32_1ch_avx2(R, Ro, w, h, k);   // اجرای هسته‌ی اسمبلی AVX2 روی R
        conv3x3_f32_1ch_avx2(G, Go, w, h, k);   // اجرای هسته‌ی اسمبلی AVX2 روی G
        conv3x3_f32_1ch_avx2(B, Bo, w, h, k);   // اجرای هسته‌ی اسمبلی AVX2 روی B
    }

    // Measure AVX2
    long long t2 = now_ns();                       // زمان شروع اندازه‌گیری AVX2/ASM
    zero_borders(Ro,w,h); zero_borders(Go,w,h); zero_borders(Bo,w,h); // مرزها قبل از اندازه‌گیری
    for (int i = 0; i < iters; i++) {              // تکرار برای دقت
        conv3x3_f32_1ch_avx2(R, Ro, w, h, k);   // اجرای ASM (R)
        conv3x3_f32_1ch_avx2(G, Go, w, h, k);   // اجرای ASM (G)
        conv3x3_f32_1ch_avx2(B, Bo, w, h, k);   // اجرای ASM (B)
    }
    long long t3 = now_ns();                       // زمان پایان اندازه‌گیری AVX2/ASM
    double avx_ms = (t3 - t2) / 1e6;               // تبدیل ns به ms

    printf("C time    : %.3f ms (iters=%d)\n", c_ms, iters);     // چاپ زمان C
    printf("AVX2 time : %.3f ms (iters=%d)\n", avx_ms, iters);   // چاپ زمان AVX2/ASM
    printf("Speedup   : %.2fx\n", c_ms / avx_ms);                // چاپ نسبت سرعت (C / AVX2)
}

int main(int argc, char** argv) {

    // --- نرمال‌سازی کرنل بر اساس مجموع ضرایب (برای کنترل روشنایی/شدت اثر) ---
    for(int i = 0; i < 3; i++){                    // جمع کردن ضرایب کرنل
        for(int j = 0; j < 3; j++){
            kernelCount += k[i][j];                // kernelCount = sum(k)
        }
    }
    if(kernelCount != 0){                          // اگر مجموع صفر نبود، نرمال‌سازی انجام بده
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                k[i][j] /= kernelCount;            // تقسیم هر ضریب بر مجموع (normalize)
            }
        }
    }

    for(int i = 0; i < 3; i++){                    // دوباره جمع می‌زنی (الان برای دیباگ/چک مجموع بعد از نرمال‌سازی)
        for(int j = 0; j < 3; j++){
            kernelCount += k[i][j];                // توجه: اینجا kernelCount دوباره accumulate میشه (ریست نشده)
        }
    }

    if (argc < 3) {                                // چک کردن آرگومان‌های ورودی
        printf("Usage: %s input.ppm output.ppm\n", argv[0]); // نحوه استفاده
        return 1;                                  // خروج با خطا
    }

    int w,h;                                       // عرض و ارتفاع تصویر
    unsigned char* rgb_u8 = read_ppm_p6(argv[1], &w, &h); // خواندن تصویر ورودی PPM
    if (!rgb_u8) return 1;                         // اگر خواندن شکست خورد

    int n = w*h;                                   // تعداد پیکسل‌ها

    // تخصیص حافظه جدا برای کانال‌ها
    float* R  = aligned_alloc(32, (size_t)n * sizeof(float)); // کانال R ورودی (aligned برای کارایی SIMD)
    float* G  = aligned_alloc(32, (size_t)n * sizeof(float)); // کانال G ورودی
    float* B  = aligned_alloc(32, (size_t)n * sizeof(float)); // کانال B ورودی

    float* Ro = aligned_alloc(32, (size_t)n * sizeof(float)); // کانال R خروجی
    float* Go = aligned_alloc(32, (size_t)n * sizeof(float)); // کانال G خروجی
    float* Bo = aligned_alloc(32, (size_t)n * sizeof(float)); // کانال B خروجی

    if (!R || !G || !B || !Ro || !Go || !Bo) {      // چک موفق بودن تخصیص‌ها
        printf("alloc failed\n");                   // پیام خطا
        return 1;                                   // خروج با خطا
    }

    aos_u8_to_soa_f32(rgb_u8, R, G, B, n);           // تبدیل RGB interleaved (u8) به کانال‌های جدا (float)

    benchmark(R, G, B, Ro, Go, Bo, w, h);            // اجرای بنچمارک و چاپ زمان‌ها

    volatile float sink = 0.0f;                      // جلوگیری از اینکه کامپایلر محاسبات رو حذف کنه
    for (int i = 0; i < w*h; i += (w*h/16 + 1)) sink += Ro[i]; // نمونه‌برداری پراکنده از خروجی برای مصرف‌کردن نتیجه
    printf("sink=%f\n", sink);                       // چاپ sink برای اطمینان از واقعی بودن محاسبه

    // #if USE_AVX2
    //     conv3x3_f32_1ch_avx2(R, Ro, w, h, k);
    //     conv3x3_f32_1ch_avx2(G, Go, w, h, k);
    //     conv3x3_f32_1ch_avx2(B, Bo, w, h, k);
    // #else
    //     conv3x3_f32_1ch(R, Ro, w, h, k);
    //     conv3x3_f32_1ch(G, Go, w, h, k);
    //     conv3x3_f32_1ch(B, Bo, w, h, k);
    // #endif

    unsigned char* out_u8 = malloc((size_t)n * 3);   // بافر خروجی RGB به صورت u8 interleaved
    soa_f32_to_aos_u8(Ro, Go, Bo, out_u8, n);         // تبدیل کانال‌های float خروجی به RGB u8 برای ذخیره در PPM

    write_ppm_p6(argv[2], out_u8, w, h);              // نوشتن فایل خروجی

    free(rgb_u8);                                     // آزاد کردن بافر ورودی
    free(R); free(G); free(B);                        // آزاد کردن کانال‌های ورودی
    free(Ro); free(Go); free(Bo);                     // آزاد کردن کانال‌های خروجی
    free(out_u8);                                     // آزاد کردن بافر خروجی
    return 0;                                         // پایان موفق
}
