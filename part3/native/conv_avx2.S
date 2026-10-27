.intel_syntax noprefix                    # استفاده از سینتکس Intel (بدون پیشوند %)
.text                                     # شروع سکشن کد
.global conv3x3_f32_1ch_avx2_asm           # این تابع برای linker قابل دیدن باشد
.type conv3x3_f32_1ch_avx2_asm, @function  # مشخص کردن نوع نماد به عنوان تابع

# ABI (SysV x86-64):
# rdi=in, rsi=out, edx=w, ecx=h, r8=k     # ترتیب آرگومان‌ها طبق قرارداد لینوکس x86-64

# Computes only vectorizable inner region:
# y = 1..h-2
# x = 1..(w-2) but only blocks of 8 floats, so up to x <= w-9
# Borders + tail must be handled by C.    # مرزها و tail بیرون (در C) مدیریت می‌شوند

conv3x3_f32_1ch_avx2_asm:
    push rbx                              # حفظ رجیسترهای callee-saved
    push r12                              # حفظ r12
    push r13                              # حفظ r13
    push r14                              # حفظ r14
    push r15                              # حفظ r15

    mov r9d, edx                          # r9d = w (عرض تصویر)
    lea r10, [r9*4]                       # r10 = stride_bytes = w*4 (هر float = 4 بایت)

    cmp edx, 10                           # اگر w < 10 باشد، حتی یک بلوک 8تایی داخل نداریم
    jl .done                              # پس از تابع خارج شو
    cmp ecx, 3                            # اگر h < 3 باشد، ناحیه داخلی نداریم
    jl .done                              # پس خارج شو

    vbroadcastss ymm8,  DWORD PTR [r8 +  0]   # k00 را در کل ymm8 پخش می‌کنیم
    vbroadcastss ymm9,  DWORD PTR [r8 +  4]   # k01
    vbroadcastss ymm10, DWORD PTR [r8 +  8]   # k02
    vbroadcastss ymm11, DWORD PTR [r8 + 12]   # k10
    vbroadcastss ymm12, DWORD PTR [r8 + 16]   # k11
    vbroadcastss ymm13, DWORD PTR [r8 + 20]   # k12
    vbroadcastss ymm14, DWORD PTR [r8 + 24]   # k20
    vbroadcastss ymm15, DWORD PTR [r8 + 28]   # k21
    vbroadcastss ymm7,  DWORD PTR [r8 + 32]   # k22 (برای صرفه‌جویی از ymm7 استفاده شده)

    mov ebx, 1                            # y = 1 (شروع ردیف‌های داخلی)

.y_loop:
    mov eax, ecx                          # eax = h
    sub eax, 1                            # eax = h-1 (حد بالا برای y)
    cmp ebx, eax                          # اگر y >= h-1
    jge .done                             # خروج (دیگه ردیف داخلی نداریم)

    mov rax, r10                          # rax = stride_bytes
    imul rax, rbx                         # rax = y*stride_bytes (آفست بایتی ردیف y)
    lea r13, [rdi + rax]                  # r13 = row1 = in + y*stride
    mov r12, r13                          # r12 = row0 موقتاً row1
    sub r12, r10                          # r12 = row0 = row1 - stride
    lea r14, [r13 + r10]                  # r14 = row2 = row1 + stride
    lea r15, [rsi + rax]                  # r15 = dst row = out + y*stride

    mov r11d, 1                           # x = 1 (شروع ستون‌های داخلی)
    mov eax, r9d                          # eax = w
    sub eax, 9                            # last = w-9 (آخرین x شروعِ بلوک 8تایی که داخل تصویر بماند)
    mov r8d, eax                          # r8d = last

.x_loop:
    cmp r11d, r8d                         # اگر x > last
    jg .x_done                            # خروج از حلقه x (tail را C حساب می‌کند)

    lea rax, [r11*4]                      # rax = x*4 (آفست بایتی ستون x)
    vxorps ymm0, ymm0, ymm0               # sum = 0 (در ymm0)

    vmovups ymm1, YMMWORD PTR [r12 + rax - 4]   # بارگذاری 8 float از row0 در (x-1)
    vfmadd231ps ymm0, ymm1, ymm8                # sum += a00 * k00
    vmovups ymm1, YMMWORD PTR [r12 + rax]       # بارگذاری row0 در (x)
    vfmadd231ps ymm0, ymm1, ymm9                # sum += a01 * k01
    vmovups ymm1, YMMWORD PTR [r12 + rax + 4]   # بارگذاری row0 در (x+1)
    vfmadd231ps ymm0, ymm1, ymm10               # sum += a02 * k02

    vmovups ymm1, YMMWORD PTR [r13 + rax - 4]   # بارگذاری row1 در (x-1)
    vfmadd231ps ymm0, ymm1, ymm11               # sum += a10 * k10
    vmovups ymm1, YMMWORD PTR [r13 + rax]       # بارگذاری row1 در (x)
    vfmadd231ps ymm0, ymm1, ymm12               # sum += a11 * k11
    vmovups ymm1, YMMWORD PTR [r13 + rax + 4]   # بارگذاری row1 در (x+1)
    vfmadd231ps ymm0, ymm1, ymm13               # sum += a12 * k12

    vmovups ymm1, YMMWORD PTR [r14 + rax - 4]   # بارگذاری row2 در (x-1)
    vfmadd231ps ymm0, ymm1, ymm14               # sum += a20 * k20
    vmovups ymm1, YMMWORD PTR [r14 + rax]       # بارگذاری row2 در (x)
    vfmadd231ps ymm0, ymm1, ymm15               # sum += a21 * k21
    vmovups ymm1, YMMWORD PTR [r14 + rax + 4]   # بارگذاری row2 در (x+1)
    vfmadd231ps ymm0, ymm1, ymm7                # sum += a22 * k22

    vmovups YMMWORD PTR [r15 + rax], ymm0       # ذخیره 8 خروجی float به out[y*w + x .. x+7]

    add r11d, 8                            # x += 8 (رفتن به بلوک بعدی)
    jmp .x_loop                            # ادامه حلقه x

.x_done:
    inc ebx                                # y++
    jmp .y_loop                            # ادامه حلقه y

.done:
    vzeroupper                             # جلوگیری از penalty بین AVX و SSE
    pop r15                                # بازگردانی رجیسترها
    pop r14                                # بازگردانی r14
    pop r13                                # بازگردانی r13
    pop r12                                # بازگردانی r12
    pop rbx                                # بازگردانی rbx
    ret                                    # بازگشت از تابع

.size conv3x3_f32_1ch_avx2_asm, .-conv3x3_f32_1ch_avx2_asm  # اندازه تابع برای assembler/linker

.section .note.GNU-stack,"",@progbits       # اعلام اینکه این فایل به stack اجرایی نیاز ندارد (برای حذف warning)
