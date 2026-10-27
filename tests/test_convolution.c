#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void conv3x3_f32_1ch(const float*, float*, int, int, const float[3][3]);
void conv3x3_f32_1ch_avx2(const float*, float*, int, int, const float[3][3]);

int main(void) {
#if defined(__x86_64__)
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) {
        fputs("AVX2 and FMA are required for this test.\n", stderr);
        return 77;
    }
#endif
    const float kernels[][3][3] = {
        {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}},
        {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}},
        {{2, 10, -3}, {-1, -6, 3}, {5, -8, -1}}
    };
    for (int w = 1; w <= 42; ++w) {
        for (int h = 1; h <= 8; ++h) {
            int n = w * h;
            float *in = malloc(n * sizeof(float));
            float *ref = malloc(n * sizeof(float));
            float *out = malloc(n * sizeof(float));
            if (!in || !ref || !out) return 2;
            for (int i = 0; i < n; ++i) in[i] = (float)((i * 17) % 29 - 14);
            for (int k = 0; k < 3; ++k) {
                for (int i = 0; i < n; ++i) ref[i] = out[i] = -12345.0f;
                conv3x3_f32_1ch(in, ref, w, h, kernels[k]);
                conv3x3_f32_1ch_avx2(in, out, w, h, kernels[k]);
                for (int i = 0; i < n; ++i) {
                    if (!isfinite(out[i]) || fabsf(ref[i] - out[i]) > 1e-4f) {
                        fprintf(stderr, "Mismatch w=%d h=%d kernel=%d pixel=%d: C=%g AVX2=%g\n",
                                w, h, k, i, ref[i], out[i]);
                        return 1;
                    }
                }
            }
            free(in); free(ref); free(out);
        }
    }
    puts("PASS: 1008 cases, including narrow images, all eight remainder sizes, and borders.");
    return 0;
}
