#pragma once                                                     // جلوگیری از دوباره‌خوانی هدر

void conv3x3_f32_1ch(                                             // کانولوشن ۳×۳ تک‌کاناله با C
    const float* in,                                              // ورودی
    float* out,                                                   // خروجی
    int w,                                                        // عرض
    int h,                                                        // ارتفاع
    const float k[3][3]                                           // کرنل
);                                                                // پایان اعلان

void conv3x3_f32_1ch_avx2(                                        // کانولوشن ۳×۳ تک‌کاناله با AVX2/اسمبلی
    const float* in,                                              // ورودی
    float* out,                                                   // خروجی
    int w,                                                        // عرض
    int h,                                                        // ارتفاع
    const float k[3][3]                                           // کرنل
);                                                                // پایان اعلان

// Assembly kernel processes complete eight-pixel interior blocks only.
void conv3x3_f32_1ch_avx2_asm(
    const float* in, float* out, int w, int h, const float k[3][3]
);
