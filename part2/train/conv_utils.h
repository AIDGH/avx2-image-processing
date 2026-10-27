#ifndef CONV_UTILS_H
#define CONV_UTILS_H

unsigned char clamp_u8(int x);

void aos_u8_to_soa_f32(
    const unsigned char* rgb,
    float* r, float* g, float* b,
    int n_pixels
);

void soa_f32_to_aos_u8(
    const float* r, const float* g, const float* b,
    unsigned char* rgb_out,
    int n_pixels
);

void conv3x3_f32_1ch(
    const float* in,
    float* out,
    int w, int h,
    const float k[3][3]
);

void conv3x3_f32_1ch_avx2(
    const float* in,
    float* out,
    int w, int h,
    const float k[3][3]
);

void conv3x3_f32_1ch_avx2_asm(
    const float* in,
    float* out,
    int w,
    int h,
    const float k[3][3]
);


#endif
