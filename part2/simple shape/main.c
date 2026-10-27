#define _POSIX_C_SOURCE 200809L  // فعال‌سازی امکانات پازیکس برای توابع زمان و فایل

#include <stdio.h>   // ورودی/خروجی
#include <stdlib.h>  // تخصیص حافظه و تبدیل‌ها
#include <string.h>  // کار با رشته‌ها
#include <ctype.h>   // isspace و ...
#include <math.h>    // توابع ریاضی مثل سینوس/کسینوس و atan2

#include "conv_utils.h" // ابزارهای کمکی پروژه (اگر لازم شد)

#define M_PI 3.14159265358979323846  // عدد پی

// ---------------- خواندن توکن از فایل PPM (با رد کردن کامنت‌ها) ----------------
static int read_token(FILE* f, char* buf, int bufsz) { // خواندن یک توکن متنی از هدر
    int c; // یک کاراکتر از فایل

    // رد کردن فاصله‌ها و خط‌های کامنت
    while (1) {
        c = fgetc(f);                         // خواندن یک کاراکتر
        if (c == EOF) return 0;               // اگر فایل تمام شد، شکست

        if (isspace(c)) continue;             // فاصله‌ها را رد کن

        if (c == '#') {                       // اگر کامنت بود
            while ((c = fgetc(f)) != EOF && c != '\n') {} // تا آخر خط برو
            continue;                         // دوباره دنبال توکن بعدی بگرد
        }

        break;                                // اینجا یعنی شروع توکن را داریم
    }

    int i = 0;              // اندیس نوشتن در بافر
    buf[i++] = (char)c;     // اولین حرف توکن

    // ادامه‌ی توکن تا رسیدن به فاصله یا کامنت
    while (i < bufsz - 1) {
        c = fgetc(f);                           // خواندن کاراکتر بعدی
        if (c == EOF) break;                    // پایان فایل
        if (isspace(c) || c == '#') {           // اگر فاصله یا شروع کامنت
            if (c == '#') ungetc(c, f);         // اگر # بود، برگردون تا دور بعد کل کامنت رد شود
            break;                              // توکن تمام شد
        }
        buf[i++] = (char)c;                     // افزودن به توکن
    }
    buf[i] = '\0';                              // پایان رشته
    return 1;                                   // موفق
}

// ---------------- خواندن تصویر PPM نوع P6 ----------------
static unsigned char* read_ppm_p6(const char* path, int* w, int* h) { // خواندن PPM دودویی
    FILE* f = fopen(path, "rb");                       // باز کردن فایل به صورت باینری
    if (!f) { perror("fopen"); return NULL; }          // اگر باز نشد، خطا

    char tok[256];                                      // بافر توکن‌ها

    if (!read_token(f, tok, sizeof(tok))) { fclose(f); return NULL; } // خواندن نوع فایل
    if (strcmp(tok, "P6") != 0) {                       // باید P6 باشد
        fprintf(stderr, "Not P6 PPM. Got magic=%s\n", tok); // گزارش خطا
        fclose(f);                                      // بستن فایل
        return NULL;                                    // شکست
    }

    if (!read_token(f, tok, sizeof(tok))) { fclose(f); return NULL; } // عرض
    *w = atoi(tok);                                     // تبدیل به عدد

    if (!read_token(f, tok, sizeof(tok))) { fclose(f); return NULL; } // ارتفاع
    *h = atoi(tok);                                     // تبدیل به عدد

    if (!read_token(f, tok, sizeof(tok))) { fclose(f); return NULL; } // بیشینه‌ی رنگ
    int maxv = atoi(tok);                               // مقدار max

    // بررسی معقول بودن مقادیر هدر
    if (*w <= 0 || *h <= 0 || maxv <= 0 || maxv > 255) {
        fprintf(stderr, "Bad header values: w=%d h=%d maxv=%d\n", *w, *h, maxv);
        fclose(f);
        return NULL;
    }

    // بعد از maxv معمولاً یک فاصله/خط‌جدید هست که قبل داده‌ی باینری می‌آید
    int c = fgetc(f);                                   // خواندن یک بایت بعد از maxv
    if (c == EOF) { fclose(f); return NULL; }           // اگر فایل تمام شد، شکست
    if (!isspace((unsigned char)c)) ungetc(c, f);       // اگر فاصله نبود، یعنی وارد داده شده‌ایم: برگردان

    // اندازه‌ی داده: سه کانال رنگی برای هر پیکسل
    size_t size = (size_t)(*w) * (size_t)(*h) * 3;      // تعداد بایت‌های داده
    unsigned char* data = (unsigned char*)malloc(size); // حافظه برای داده
    if (!data) { fclose(f); return NULL; }              // اگر حافظه نداد، شکست

    size_t rd = fread(data, 1, size, f);                // خواندن داده‌ی تصویر
    fclose(f);                                          // بستن فایل

    if (rd != size) {                                   // اگر کم خوانده شد
        fprintf(stderr, "PPM read incomplete: got %zu, expected %zu\n", rd, size);
        free(data);
        return NULL;
    }
    return data;                                        // موفق
}

// ---------------- نوشتن تصویر PPM نوع P6 ----------------
static int write_ppm_p6(const char* path, const unsigned char* data, int w, int h) { // نوشتن PPM دودویی
    FILE* f = fopen(path, "wb");                        // باز کردن خروجی
    if (!f) { perror("fopen"); return 0; }              // خطا در باز کردن

    fprintf(f, "P6\n%d %d\n255\n", w, h);                // هدر P6 با max=255
    fwrite(data, 1, (size_t)w * (size_t)h * 3, f);       // نوشتن داده‌ی تصویر
    fclose(f);                                          // بستن فایل
    return 1;                                           // موفق
}

// ---------------- کشیدن کادر قرمز دور شکل ----------------
static void draw_rect_rgb(unsigned char* rgb, int w, int h, int x0, int y0, int x1, int y1) { // رسم کادر
    if (x0 < 0) x0 = 0;                     // محدود کردن به داخل تصویر
    if (y0 < 0) y0 = 0;                     // محدود کردن به داخل تصویر
    if (x1 >= w) x1 = w - 1;                // محدود کردن به داخل تصویر
    if (y1 >= h) y1 = h - 1;                // محدود کردن به داخل تصویر
    if (x0 > x1 || y0 > y1) return;         // اگر کادر نامعتبر بود، کاری نکن

    // ضخامت کادر = ۲ پیکسل
    for (int t = 0; t < 2; t++) {           // لایه‌ی ضخامت
        int yt0 = y0 + t, yt1 = y1 - t;     // بالا و پایین
        int xt0 = x0 + t, xt1 = x1 - t;     // چپ و راست

        // خط بالا
        if (yt0 >= 0 && yt0 < h) {
            for (int x = xt0; x <= xt1; x++) {
                int idx = (yt0*w + x)*3;    // آدرس پیکسل در آرایه‌ی RGB
                rgb[idx+0] = 255;           // قرمز
                rgb[idx+1] = 0;             // سبز
                rgb[idx+2] = 0;             // آبی
            }
        }
        // خط پایین
        if (yt1 >= 0 && yt1 < h) {
            for (int x = xt0; x <= xt1; x++) {
                int idx = (yt1*w + x)*3;
                rgb[idx+0] = 255; rgb[idx+1] = 0; rgb[idx+2] = 0;
            }
        }
        // خط چپ
        if (xt0 >= 0 && xt0 < w) {
            for (int y = yt0; y <= yt1; y++) {
                int idx = (y*w + xt0)*3;
                rgb[idx+0] = 255; rgb[idx+1] = 0; rgb[idx+2] = 0;
            }
        }
        // خط راست
        if (xt1 >= 0 && xt1 < w) {
            for (int y = yt0; y <= yt1; y++) {
                int idx = (y*w + xt1)*3;
                rgb[idx+0] = 255; rgb[idx+1] = 0; rgb[idx+2] = 0;
            }
        }
    }
}

// ---------------- تشخیص اینکه پیکسل روی مرز شکل است یا نه ----------------
static inline int is_boundary(const unsigned char* fill, int w, int h, int x, int y) { // مرزی بودن
    int idx = y*w + x;                                 // اندیس در آرایه
    if (!fill[idx]) return 0;                          // اگر جزو شکل نیست، مرز هم نیست
    if (x <= 0 || x >= w-1 || y <= 0 || y >= h-1) return 1; // کنار تصویر را مرز حساب کن
    if (!fill[idx-1] || !fill[idx+1] || !fill[idx-w] || !fill[idx+w]) return 1; // اگر همسایه‌ی چهارجهت خالی بود
    return 0;                                          // در غیر این صورت مرز نیست
}

// ---------------- تصمیم‌گیری نوع شکل با چند ویژگی ----------------
static const char* classify_shape( // تعیین برچسب نوع شکل
    float circularity,             // دایره‌ای بودن تقریبی
    float aspect_rot_err,           // خطای نسبت طول و عرض در جعبه‌ی چرخیده
    float aspect_axis_err,          // خطای نسبت طول و عرض در جعبه‌ی معمولی
    float angle_deg,                // زاویه‌ی محور اصلی شکل
    float fill_ratio_axis,          // نسبت پرشدگی داخل جعبه‌ی معمولی
    int approx_vertices             // تعداد گوشه‌ی تقریبی
) {
    if(aspect_axis_err == 1000){
        return "UNKNOWN";
    }
    // دایره: دایره‌ای بودن زیاد
    if (circularity > 0.85f) return "CIRCLE"; // دایره

    // چندضلعی‌ها اگر تعداد گوشه مشخص شد
    if (approx_vertices == 3) return "TRIANGLE"; // مثلث
    if (approx_vertices == 5) return "PENTAGON"; // پنج‌ضلعی

    // خانواده‌ی چهارضلعی‌ها (مربع/مستطیل/لوزی)
    if (approx_vertices == 4 || approx_vertices == 0) { // اگر ۴ گوشه یا نامشخص
        // مربع/لوزی: جعبه‌ی چرخیده باید نزدیک مربع باشد
        if (aspect_rot_err < 0.12f) {
            float a = fmodf(fabsf(angle_deg), 90.0f); // زاویه را به بازه‌ی ۰ تا ۹۰ بیاور
            float near45 = fabsf(a - 45.0f);         // فاصله از ۴۵ درجه

            // لوزی: زاویه نزدیک ۴۵ و جعبه‌ی معمولی خیلی پر نشده باشد
            if (fill_ratio_axis < 0.75f && near45 < 15.0f) return "DIAMOND"; // لوزی
            return "SQUARE"; // مربع
        }
        return "RECTANGLE"; // مستطیل
    }

    return "UNKNOWN"; // نامشخص
}

// ---------------- تخمین تعداد گوشه با روش شعاعی روی نقاط مرزی ----------------
static int estimate_vertices_radial(const int* bx, const int* by, int bcnt, float cx, float cy) { // تخمین گوشه‌ها
    if (bcnt < 60) return 0;                       // اگر مرز خیلی کم است، بی‌خیال

    int S = 180;                                   // تعداد بازه‌های زاویه‌ای
    float* maxr = (float*)calloc((size_t)S, sizeof(float)); // بیشینه شعاع در هر زاویه
    if (!maxr) return 0;                           // اگر حافظه نداد، شکست

    // برای هر نقطه مرزی: زاویه و شعاع را حساب کن و بیشینه را نگه دار
    for (int i = 0; i < bcnt; i++) {
        float dx = (float)bx[i] - cx;              // اختلاف x از مرکز
        float dy = (float)by[i] - cy;              // اختلاف y از مرکز
        float ang = atan2f(dy, dx);                // زاویه در بازه‌ی -پی تا +پی
        float t = (ang + (float)M_PI) * (float)S / (2.0f*(float)M_PI); // نگاشت زاویه به ۰..S
        int k = (int)t;                            // شاخص بازه
        if (k < 0) k = 0;                          // محدودسازی
        if (k >= S) k = S-1;                       // محدودسازی
        float r = sqrtf(dx*dx + dy*dy);            // فاصله از مرکز
        if (r > maxr[k]) maxr[k] = r;              // ثبت بیشینه شعاع در آن زاویه
    }

    // نرم‌سازی ساده برای کاهش نویز
    float* sm = (float*)calloc((size_t)S, sizeof(float)); // آرایه‌ی نرم‌شده
    if (!sm) { free(maxr); return 0; }             // اگر حافظه نداد، شکست
    for (int i = 0; i < S; i++) {
        float a = maxr[(i-1+S)%S];                 // مقدار قبلی
        float b = maxr[i];                         // مقدار فعلی
        float c = maxr[(i+1)%S];                   // مقدار بعدی
        sm[i] = (a + 2*b + c) * 0.25f;             // فیلتر سه‌تایی
    }

    // میانگین شعاع
    float mean = 0.0f;                             // جمع برای میانگین
    for (int i = 0; i < S; i++) mean += sm[i];     // جمع
    mean /= (float)S;                              // میانگین

    // شمارش قله‌ها: هر قله یک گوشه‌ی احتمالی
    int peaks = 0;                                 // تعداد قله‌ها
    for (int i = 0; i < S; i++) {
        float prev = sm[(i-1+S)%S];                // قبل
        float cur  = sm[i];                        // فعلی
        float next = sm[(i+1)%S];                  // بعد
        if (cur > prev && cur > next && cur > mean * 1.08f) peaks++; // قله‌ی قابل توجه
    }

    free(sm);                                      // آزادسازی
    free(maxr);                                    // آزادسازی

    // محدود کردن به بازه‌ی منطقی
    if (peaks >= 2 && peaks <= 10) return peaks;   // خروجی معتبر
    return 0;                                      // در غیر این صورت نامطمئن
}

int main(int argc, char** argv) { // نقطه شروع برنامه
    if (argc < 2) { // اگر ورودی نداریم
        printf("Usage: %s input.ppm [bin_thr]\n", argv[0]); // راهنمای استفاده
        printf("Example: %s shapes.ppm 128\n", argv[0]);    // نمونه
        return 1; // خروج با خطا
    }

    int bin_thr = 128;                         // آستانه‌ی دودویی کردن تصویر خاکستری
    if (argc >= 3) bin_thr = atoi(argv[2]);    // اگر کاربر مقدار داد، استفاده کن
    if (bin_thr < 1) bin_thr = 1;              // محدودسازی پایین
    if (bin_thr > 254) bin_thr = 254;          // محدودسازی بالا

    int w, h;                                  // عرض و ارتفاع
    unsigned char* rgb = read_ppm_p6(argv[1], &w, &h); // خواندن تصویر
    if (!rgb) return 1;                        // اگر نشد، خروج

    int n = w*h;                               // تعداد کل پیکسل‌ها

    float* gray = (float*)malloc((size_t)n * sizeof(float)); // تصویر خاکستری
    unsigned char* fill = (unsigned char*)malloc((size_t)n); // ماسک پر بودن شکل‌ها (۰ یا ۱)
    unsigned char* vis  = (unsigned char*)calloc((size_t)n, 1); // بازدید شده‌ها برای پیمایش

    int* qx = (int*)malloc((size_t)n * sizeof(int)); // صف x برای پیمایش
    int* qy = (int*)malloc((size_t)n * sizeof(int)); // صف y برای پیمایش

    int* bx = (int*)malloc((size_t)n * sizeof(int)); // ذخیره نقاط مرزی x
    int* by = (int*)malloc((size_t)n * sizeof(int)); // ذخیره نقاط مرزی y

    // بررسی موفق بودن تخصیص‌ها
    if (!gray || !fill || !vis || !qx || !qy || !bx || !by) {
        fprintf(stderr, "alloc failed\n");                 // پیام خطا
        free(rgb);                                         // آزادسازی
        free(gray); free(fill); free(vis); free(qx); free(qy); free(bx); free(by); // آزادسازی
        return 1;                                          // خروج
    }

    // تبدیل RGB به خاکستری و ساخت ماسک پرشدگی
    for (int i = 0; i < n; i++) {                          // روی همه پیکسل‌ها
        unsigned char R = rgb[i*3+0];                      // کانال قرمز
        unsigned char G = rgb[i*3+1];                      // کانال سبز
        unsigned char B = rgb[i*3+2];                      // کانال آبی
        float g = 0.299f*(float)R + 0.587f*(float)G + 0.114f*(float)B; // فرمول خاکستری
        gray[i] = g;                                       // ذخیره خاکستری
        // --- محاسبه میانگین روشنایی حاشیه برای تشخیص رنگ پس‌زمینه ---
        // double border_sum = 0.0;        // جمع روشنایی حاشیه
        // int border_cnt = 0;             // تعداد پیکسل‌های حاشیه

        // for (int x = 0; x < w; x++) {   // بالا و پایین
        //     border_sum += gray[x];
        //     border_sum += gray[(h-1)*w + x];
        //     border_cnt += 2;
        // }

        // for (int y = 1; y < h-1; y++) { // چپ و راست
        //     border_sum += gray[y*w];
        //     border_sum += gray[y*w + (w-1)];
        //     border_cnt += 2;
        // }

        // double border_mean = border_sum / (double)border_cnt;  // میانگین روشنایی حاشیه

        // // اگر پس‌زمینه روشن بود → شکل تیره است
        // int invert = (border_mean > 127.0) ? 1 : 0;

        // for (int i = 0; i < n; i++) {
        //     if (!invert)
        //         fill[i] = (gray[i] >= (float)bin_thr) ? 1 : 0;
        //     else
        //         fill[i] = (gray[i] < (float)bin_thr) ? 1 : 0;
        // }
        fill[i] = (g >= (float)bin_thr) ? 1 : 0;           // اگر روشن‌تر از آستانه بود، جزو شکل
    }

    // ساخت خروجی تصویر برای کشیدن کادرها
    unsigned char* boxed = (unsigned char*)malloc((size_t)n * 3); // تصویر خروجی
    if (!boxed) {                                              // اگر حافظه نداد
        fprintf(stderr, "alloc boxed failed\n");                // پیام
        free(rgb); free(gray); free(fill); free(vis); free(qx); free(qy); free(bx); free(by); // آزادسازی
        return 1;                                               // خروج
    }
    memcpy(boxed, rgb, (size_t)n * 3);                          // کپی تصویر اصلی

    unsigned char* edges = (unsigned char*)calloc((size_t)n * 3, 1); // ساخت تصویر لبه‌ها (RGB) و صفر کردنش (پس‌زمینه سیاه)
    if (!edges) {                                                    // اگر حافظه نداد
        fprintf(stderr, "alloc edges failed\n");                      // پیام خطا
        free(boxed);                                                  // آزادسازی boxed
        free(rgb); free(gray); free(fill); free(vis);                 // آزادسازی بقیه
        free(qx); free(qy); free(bx); free(by);                       // آزادسازی آرایه‌ها
        return 1;                                                     // خروج
    }

    // پیمایش جزءهای همبند روی ماسک پرشدگی
    int comp_id = 0;                                            // شماره‌ی جزءها

    printf("bin_thr=%d\n", bin_thr);                             // چاپ آستانه

    for (int y0 = 1; y0 < h-1; y0++) {                           // پیمایش سطرها (بدون حاشیه)
        for (int x0 = 1; x0 < w-1; x0++) {                       // پیمایش ستون‌ها (بدون حاشیه)
            int idx0 = y0*w + x0;                                // اندیس پیکسل
            if (!fill[idx0] || vis[idx0]) continue;              // اگر جزو شکل نیست یا قبلاً دیده شده، رد کن

            comp_id++;                                           // شروع جزء جدید
            int head = 0, tail = 0;                              // ابتدا و انتهای صف
            qx[tail] = x0; qy[tail] = y0; tail++;                // افزودن نقطه شروع به صف
            vis[idx0] = 1;                                       // علامت‌گذاری بازدید شده

            int minx = x0, maxx = x0, miny = y0, maxy = y0;      // جعبه‌ی محدودکننده‌ی معمولی

            // جمع‌ها برای محاسبه مرکز و کوواریانس (برای زاویه)
            double sumx = 0.0, sumy = 0.0;                       // جمع مختصات
            double sumxx = 0.0, sumyy = 0.0, sumxy = 0.0;        // جمع‌های درجه دوم

            int area = 0;                                        // مساحت (تعداد پیکسل‌های پر)
            int bcnt = 0;                                        // تعداد پیکسل‌های مرزی

            // پیمایش پهنایی (صف) برای جمع کردن کل جزء
            while (head < tail) {                                // تا زمانی که صف خالی نشده
                int x = qx[head];                                // برداشت x
                int y = qy[head];                                // برداشت y
                head++;                                          // جلو بردن سر صف

                area++;                                          // افزایش مساحت

                // به‌روزرسانی جعبه‌ی محدودکننده
                if (x < minx) minx = x;                          // کمینه x
                if (x > maxx) maxx = x;                          // بیشینه x
                if (y < miny) miny = y;                          // کمینه y
                if (y > maxy) maxy = y;                          // بیشینه y

                // به‌روزرسانی جمع‌ها برای محاسبه‌ی مرکز و زاویه
                sumx += x; sumy += y;                            // جمع مختصات
                sumxx += (double)x*(double)x;                    // جمع x^2
                sumyy += (double)y*(double)y;                    // جمع y^2
                sumxy += (double)x*(double)y;                    // جمع x*y

                // اگر روی مرز بود ذخیره کن
                if (is_boundary(fill, w, h, x, y)) {             // بررسی مرزی بودن
                    bx[bcnt] = x;                                // ذخیره x مرز
                    by[bcnt] = y;                                // ذخیره y مرز
                    bcnt++;                                      // افزایش شمار مرز
                    int e = (y*w + x)*3;                          // آدرس پیکسل متناظر در تصویر edges
                    edges[e+0] = 255;                             // لبه را سفید کن (R)
                    edges[e+1] = 255;                             // لبه را سفید کن (G)
                    edges[e+2] = 255;                             // لبه را سفید کن (B)
                }

                // همسایه‌های چهارجهت برای جزءهای پر
                const int nx[4] = { x-1, x+1, x, x };            // x همسایه‌ها
                const int ny[4] = { y, y, y-1, y+1 };            // y همسایه‌ها
                for (int k = 0; k < 4; k++) {                    // روی ۴ همسایه
                    int xx = nx[k], yy = ny[k];                  // مختصات همسایه
                    if (xx <= 0 || xx >= w-1 || yy <= 0 || yy >= h-1) continue; // خارج از محدوده
                    int ni = yy*w + xx;                          // اندیس همسایه
                    if (fill[ni] && !vis[ni]) {                  // اگر پر و دیده نشده
                        vis[ni] = 1;                             // علامت‌گذاری
                        qx[tail] = xx;                           // افزودن به صف
                        qy[tail] = yy;                           // افزودن به صف
                        tail++;                                  // انتهای صف جلو
                    }
                }
            }

            // حذف لکه‌های کوچک
            if (area < 200) continue;                             // اگر خیلی کوچک بود، رد کن

            // جعبه‌ی معمولی و نسبت پرشدگی داخل آن
            int bw = maxx - minx + 1;                             // عرض جعبه
            int bh = maxy - miny + 1;                             // ارتفاع جعبه
            float axis_aspect_err = 1.0f;                         // خطای مربعی بودن در جعبه‌ی معمولی
            {
                int mx = (bw > bh) ? bw : bh;                     // بزرگ‌تر
                int mn = (bw < bh) ? bw : bh;                     // کوچک‌تر
                axis_aspect_err = (mx > 0) ? (1.0f - (float)mn / (float)mx) : 1.0f; // نزدیک صفر یعنی مربع
            }
            float fill_ratio_axis = (float)area / (float)(bw*bh); // نسبت پرشدگی (۱ یعنی کاملاً پر)

            // محاسبه مرکز جرم
            float cx = (float)(sumx / (double)area);              // مرکز x
            float cy = (float)(sumy / (double)area);              // مرکز y

            // محاسبه کوواریانس برای زاویه‌ی محور اصلی
            double meanx = sumx / (double)area;                   // میانگین x
            double meany = sumy / (double)area;                   // میانگین y
            double covxx = (sumxx / (double)area) - meanx*meanx;  // کوواریانس xx
            double covyy = (sumyy / (double)area) - meany*meany;  // کوواریانس yy
            double covxy = (sumxy / (double)area) - meanx*meany;  // کوواریانس xy

            // زاویه‌ی محور اصلی: فرمول بسته برای بردار ویژه
            double theta = 0.5 * atan2(2.0*covxy, (covxx - covyy)); // زاویه بر حسب رادیان
            float angle_deg = (float)(theta * 180.0 / M_PI);        // زاویه بر حسب درجه

            // محاسبه جعبه‌ی چرخیده با چرخاندن نقاط مرزی حول مرکز
            float ct = cosf((float)theta);                        // کسینوس زاویه
            float st = sinf((float)theta);                        // سینوس زاویه

            float rminx =  1e30f, rmaxx = -1e30f;                 // کمینه/بیشینه x در دستگاه چرخیده
            float rminy =  1e30f, rmaxy = -1e30f;                 // کمینه/بیشینه y در دستگاه چرخیده

            for (int i = 0; i < bcnt; i++) {                      // روی نقاط مرزی
                float dx = (float)bx[i] - cx;                     // انتقال به مرکز
                float dy = (float)by[i] - cy;                     // انتقال به مرکز
                float rx =  dx*ct + dy*st;                        // چرخش x
                float ry = -dx*st + dy*ct;                        // چرخش y
                if (rx < rminx) rminx = rx;                       // کمینه rx
                if (rx > rmaxx) rmaxx = rx;                       // بیشینه rx
                if (ry < rminy) rminy = ry;                       // کمینه ry
                if (ry > rmaxy) rmaxy = ry;                       // بیشینه ry
            }

            float rbw = rmaxx - rminx;                            // عرض جعبه‌ی چرخیده
            float rbh = rmaxy - rminy;                            // ارتفاع جعبه‌ی چرخیده
            float rot_aspect_err = 1.0f;                          // خطای مربعی بودن در جعبه‌ی چرخیده
            {
                float mx = (rbw > rbh) ? rbw : rbh;               // بزرگ‌تر
                float mn = (rbw < rbh) ? rbw : rbh;               // کوچک‌تر
                rot_aspect_err = (mx > 1e-6f) ? (1.0f - mn/mx) : 1.0f; // نزدیک صفر یعنی مربع
            }

            // محیط تقریبی: تعداد پیکسل‌های مرزی (تقریبی است)
            float perimeter = (float)bcnt;                        // محیط تقریبی
            float circularity = 0.0f;                             // دایره‌ای بودن
            if (perimeter > 1e-3f) {                              // جلوگیری از تقسیم بر صفر
                circularity = (float)(4.0 * M_PI) * (float)area / (perimeter*perimeter); // فرمول دایره‌ای بودن
            }

            // تخمین تعداد گوشه با روش شعاعی (تقریبی)
            int approx_vertices = estimate_vertices_radial(bx, by, bcnt, cx, cy); // تعداد گوشه تقریبی

            // تعیین نوع شکل با قوانین ساده
            const char* type = classify_shape(                    // تعیین نوع
                circularity,                                      // دایره‌ای بودن
                rot_aspect_err,                                   // خطای جعبه چرخیده
                axis_aspect_err,                                  // خطای جعبه معمولی
                angle_deg,                                        // زاویه
                fill_ratio_axis,                                  // پرشدگی
                approx_vertices                                   // تعداد گوشه
            );

            int centerx = (minx + maxx) / 2;                      // مرکز جعبه معمولی x
            int centery = (miny + maxy) / 2;                      // مرکز جعبه معمولی y

            // چاپ اطلاعات هر شکل
            printf("[#%d] type=%-9s center=(%d,%d) area=%d per~%.0f circ=%.3f "
                   "bbox=%dx%d fill=%.2f angle=%.1f rot_err=%.3f vtx~%d\n",
                   comp_id, type, centerx, centery, area, perimeter, circularity,
                   bw, bh, fill_ratio_axis, angle_deg, rot_aspect_err, approx_vertices);

            // کشیدن کادر روی تصویر خروجی
            draw_rect_rgb(boxed, w, h, minx, miny, maxx, maxy);   // کادر قرمز دور شکل
        }
    }

    // ذخیره خروجی تصویر دارای کادرها
    write_ppm_p6("boxed.ppm", boxed, w, h);                       // نوشتن boxed
    write_ppm_p6("edges.ppm", edges, w, h);                       // نوشتن edges

    // آزادسازی همه حافظه‌ها
    free(boxed);                                                  // آزادسازی تصویر خروجی
    free(bx); free(by);                                           // آزادسازی نقاط مرزی
    free(qx); free(qy);                                           // آزادسازی صف
    free(vis);                                                    // آزادسازی بازدیدها
    free(fill);                                                   // آزادسازی ماسک پرشدگی
    free(gray);                                                   // آزادسازی خاکستری
    free(rgb);                                                    // آزادسازی تصویر ورودی
    return 0;                                                     // پایان موفق
}
