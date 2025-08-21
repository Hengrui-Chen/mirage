#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <assert.h>
#include <stdio.h>
#include "linear.cuh"  // 假设包含 linear_kernel 和 linear_kernel_fp8_weight
#include <algorithm> // std::max
#include <cstdio>

using bfloat16 = kernel::bfloat16;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
constexpr int MAX_SHARE_MEMORY_SIZE = 224 * 1024;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
constexpr int MAX_SHARE_MEMORY_SIZE = 96 * 1024;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
constexpr int MAX_SHARE_MEMORY_SIZE = 160 * 1024;
#else
constexpr int MAX_SHARE_MEMORY_SIZE = 96 * 1024;
#endif

// ===== 测试尺寸 =====
constexpr int BATCH_SIZE     = 1;
constexpr int OUTPUT_SIZE    = 64;  // N
constexpr int REDUCTION_SIZE = 1024;  // K

// ===== 工具宏 =====
#define CHECK_CUDA(expr) do {                                     \
  cudaError_t _e = (expr);                                        \
  if (_e != cudaSuccess) {                                        \
    fprintf(stderr, "CUDA error %s at %s:%d : %s\n",              \
            #expr, __FILE__, __LINE__, cudaGetErrorString(_e));   \
    exit(1);                                                      \
  }                                                               \
} while(0)

// ---- Simple timing helpers ---------------------------------------------------
struct BenchResult {
    float avg_ms, std_ms;
};

template <typename Launch>
static inline BenchResult time_kernel(int warmup, int iters, Launch launch) {
    CHECK_CUDA(cudaDeviceSynchronize());

    for (int i = 0; i < warmup; ++i) { launch(); }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> times; times.reserve(iters);
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < iters; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        launch();
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaDeviceSynchronize());

    double sum = 0.0, sq = 0.0;
    for (float t : times) { sum += t; sq += (double)t * t; }
    double avg = sum / iters;
    double var = std::max(0.0, sq / iters - avg * avg);
    return { (float)avg, (float)std::sqrt(var) };
}


// GEMM/GEMV FLOPs (multiply+add = 2 ops)
static inline double gemm_flops(int M, int N, int K) {
    return 2.0 * (double)M * (double)N * (double)K;
}

// A rough lower-bound memory traffic (bytes) for one matmul+residual write.
// This is *not* exact (doesn't include all internal reads/writes), but helpful for GB/s.
static inline double bytes_moved_baseline_bf16(int M, int N, int K) {
    const double sz_bf16 = 2.0;
    return M*K*sz_bf16           // input
         + K*N*sz_bf16           // weight
         + M*N*sz_bf16           // residual
         + M*N*sz_bf16;          // output
}

static inline double bytes_moved_fp8_scale(int M, int N, int K, int BK=128, int BN=128) {
    const double sz_bf16 = 2.0, sz_fp8 = 1.0;
    const int nBK = (K + BK - 1) / BK;
    const int nBN = (N + BN - 1) / BN;
    return M*K*sz_bf16           // input
         + K*N*sz_fp8            // weight (fp8)
         + (nBK*nBN)*sz_bf16     // block scales
         + M*N*sz_bf16           // residual
         + M*N*sz_bf16;          // output
}




// ===== 简易随机填充 =====
template<typename T>
void fill_random(std::vector<T>& vec, float scale=100.0f) {
    for (auto& v : vec) {
        v = static_cast<T>((static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2 * scale);
        // v = static_cast<T>(1); // for debug
    }
}

template<typename T>
void fill_random_grouped(std::vector<T>& vec, int K, int N, float scale=100.0f,
    int BK=128, int BN=128) {
    // for (auto& v : vec) {
    //     v = static_cast<T>((static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2 * scale);
    //     // v = static_cast<T>(1); // for debug
    // }
    const int nBK = (K + BK - 1) / BK;
    const int nBN = (N + BN - 1) / BN;
    std::vector<bfloat16> scales(nBK * nBN);

    for (int br = 0; br < nBK; ++br) {
        int k0 = br * BK;
        int k1 = std::min(k0 + BK, K);
        for (int bc = 0; bc < nBN; ++bc) {
            int n0 = bc * BN;
            int n1 = std::min(n0 + BN, N);
            float group_scale = (static_cast<float>(rand()) / RAND_MAX) * scale + 1.0f; // [1, 1+scale]

            for (int k = k0; k < k1; ++k) {
                for (int n = n0; n < n1; ++n) {
                    vec[k * N + n] = static_cast<T>((static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2 * group_scale);
                }
            }
        }
    }
    return;
}



// #include <math.h>
// #include <stdint.h>
// #include <float.h>

// ---------- FP32 -> FP8 (E4M3, no +/-Inf; 0x?F=exp=1111 allowed; 0x?F=NaN only when mant=111) ----------
static inline uint8_t float_to_fp8_e4m3(float x) {
    // NaN -> quiet NaN pattern: exp=1111, mant=111, sign ignored (choose +)
    if (isnan(x)) return 0x7F; // 0b0_1111_111
    // ±Inf -> 饱和到最大有限值 ±448 ：exp=1111, mant=110
    if (isinf(x)) return ((signbit(x) ? 1 : 0) << 7) | (0xF << 3) | 0x6;

    int s = signbit(x) ? 1 : 0;
    float a = fabsf(x);

    // 0 直接返回 0
    if (a == 0.0f) return 0;

    // 最大有限值阈值
    const float MAX_FINITE = 448.0f; // exp=15, mant=6
    if (a >= MAX_FINITE) {
        // 饱和到最大有限可编码值（不是 Inf）
        return (uint8_t)((s << 7) | (0xF << 3) | 0x6);
    }

    // 使用 frexpf: a = m * 2^e, m ∈ [0.5,1)
    int e;
    float m = frexpf(a, &e); // 0.5 <= m < 1
    // 目标无偏指数（E4M3 偏置 7），frexpf 的 m 在 [0.5,1)，所以 E = e + (bias-1) = e + 6
    int E = e + 6;

    // 处理规范化与次正规
    if (E <= 0) {
        // 次正规：val ≈ (mant/8) * 2^(1-7) = mant * 2^-9
        // 求 mant ≈ a * 2^9，做 nearest-even
        float mant_f = ldexpf(a, 9);            // a * 2^9
        int mant = (int)lrintf(mant_f);         // nearest-even
        if (mant > 7) mant = 7;
        if (mant == 0) return 0;                // 太小当作 0
        return (uint8_t)((s << 7) | (0 << 3) | (mant & 0x7));
    } else {
        // 规范化：先按 frexpf 的 m 求 mant
        // m ∈ [0.5,1) ; f = (m-0.5)/0.5 = 2*m - 1 ∈ [0,1)
        float f = 2.0f * m - 1.0f;
        float mant_f = f * 8.0f;                // 目标 3 位
        int mant = (int)lrintf(mant_f);         // nearest-even

        // 舍入可能导致 mant==8，需要进位
        if (mant == 8) {
            mant = 0;
            E += 1;
        }

        // 指数上溢：饱和到最大有限（E=15, mant=6）
        if (E > 0xF || (E == 0xF && mant == 7)) {
            // E4M3 无 Inf；exp=1111 仍可用，但 mant=111 保留给 NaN
            // 因此饱和为 mant=110（=6）
            return (uint8_t)((s << 7) | (0xF << 3) | 0x6);
        }

        // 正常范围内直接编码
        return (uint8_t)((s << 7) | ((E & 0xF) << 3) | (mant & 0x7));
    }
}

uint8_t float_to_fp8(float val) {
    return float_to_fp8_e4m3(val);
}


// ---------- FP8 -> FP32 ----------
float fp8_to_float(uint8_t v, bool e5m2 /* = false */) {
    int sign = (v & 0x80) ? -1 : 1;

    if (e5m2) {
        // E5M2：保持你原实现
        int exponent = (v >> 2) & 0x1F;
        int mantissa = v & 0x03;
        if (exponent == 0) {
            // subnormal
            return sign * ldexpf((float)mantissa, -2 - 14);
        } else if (exponent == 0x1F) {
            // E5M2 有 Inf/NaN
            return sign * (mantissa ? NAN : INFINITY);
        }
        return sign * (1.0f + mantissa / 4.0f) * ldexpf(1.0f, exponent - 15);
    } else {
        // E4M3（新定义）
        int exponent = (v >> 3) & 0x0F; // 4-bit exponent
        int mantissa = v & 0x07;        // 3-bit mantissa

        if (exponent == 0) {
            // 次正规：mant * 2^-9
            if (mantissa == 0) return sign * 0.0f;
            return sign * ldexpf((float)mantissa, -9);
        } else if (exponent == 0x0F) {
            // exp=1111：仍可为规格化；唯独 mant=111 是 NaN（无 Inf）
            if (mantissa == 0x7) {
                return NAN;
            } else {
                // 规范化，指数当作 15 使用
                return sign * (1.0f + mantissa / 8.0f) * ldexpf(1.0f, 15 - 7);
            }
        } else {
            // 普通规范化
            return sign * (1.0f + mantissa / 8.0f) * ldexpf(1.0f, exponent - 7);
        }
    }
}

template<typename T>
void fill_random_grouped2(std::vector<T>& vec, int K, int N, float scale=100.0f,
    int BK=128, int BN=128) {
    // for (auto& v : vec) {
    //     v = static_cast<T>((static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2 * scale);
    //     // v = static_cast<T>(1); // for debug
    // }
    const int nBK = (K + BK - 1) / BK;
    const int nBN = (N + BN - 1) / BN;
    // std::vector<bfloat16> scales(nBK * nBN);

    // for (int br = 0; br < nBK; ++br) {
    //     int k0 = br * BK;
    //     int k1 = std::min(k0 + BK, K);
    //     for (int bc = 0; bc < nBN; ++bc) {
    //         int n0 = bc * BN;
    //         int n1 = std::min(n0 + BN, N);
    //         float group_scale = (static_cast<float>(rand()) / RAND_MAX) * scale + 1.0f; // [1, 1+scale]

    //         for (int k = k0; k < k1; ++k) {
    //             for (int n = n0; n < n1; ++n) {
    //                 // float t = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2 * group_scale;
    //                 // generate a fp8 random value
    //                 uint8_t fp8_value = float_to_fp8( (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2 * 440.0f );
    //                 vec[k * N + n] = static_cast<T>( fp8_to_float(fp8_value, false) / 440.0f * group_scale); // scale to roughly [-group_scale, group_scale]
    //             }
    //         }
    //     }
    // }
    for (int bc = 0; bc < nBN; ++bc) {
        const int n0 = bc * BN;
        const int n1 = std::min(n0 + BN, N);
        for (int br = 0; br < nBK; ++br) {
            const int k0 = br * BK;
            const int k1 = std::min(k0 + BK, K);
            float group_scale = (static_cast<float>(rand()) / RAND_MAX) * scale + 1.0f; // [1, 1+scale]

            // 列主序访问：先按列 n，再按行 k，索引 base = n*K
            for (int n = n0; n < n1; ++n) {
                const int base = n * K; // column-major
                for (int k = k0; k < k1; ++k) {
                    // generate a fp8 random value
                    uint8_t fp8_value = float_to_fp8( (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2 * 440.0f );
                    vec[base + k] = static_cast<T>( fp8_to_float(fp8_value, false) / 440.0f * group_scale); // scale to roughly [-group_scale, group_scale]
                }
            }
        }
    }
    return;
}



// // host 上的 FP16 -> FP8 (E4M3) 量化
// uint8_t float_to_fp8(float val) {
//     int sign = val < 0 ? 1 : 0;
//     val = fabsf(val);

//     if (val == 0) return 0;

//     int exp;
//     float mant = frexpf(val, &exp); // val = mant * 2^(exp)
//     exp += 6; // 对应 E4M3 偏移量

//     if (exp <= 0) { // 次正规数
//         int m = (int)(mant * (1 << (exp + 2))); 
//         return (sign << 7) | (0 << 3) | (m & 0x7);
//     } else if (exp >= 0xF) { // 溢出
//         return (sign << 7) | (0xF << 3); // Inf
//     } else { 
//         int m = (int)((mant - 0.5f) * (1 << 4)); 
//         return (sign << 7) | ((exp & 0xF) << 3) | (m & 0x7);
//     }
// }


// float fp8_to_float(uint8_t val, bool e5m2 = false) {
//     // Sign extraction
//     int sign = (val & 0x80) ? -1 : 1; // MSB is sign bit
//     int exponent, mantissa;

//     if (e5m2) {
//         // FP8 E5M2: 1 sign, 5 exponent, 2 mantissa
//         exponent = (val >> 2) & 0x1F;   // bits 2-6
//         mantissa = val & 0x03;          // bits 0-1
//         if (exponent == 0) {
//             // Subnormal
//             return sign * ldexpf((float)mantissa, -2 - 14); 
//         } else if (exponent == 0x1F) {
//             return sign * (mantissa ? NAN : INFINITY); 
//         }
//         return sign * (1.0f + mantissa / 4.0f) * ldexpf(1.0f, exponent - 15);
//     } 
//     else {
//         // FP8 E4M3: 1 sign, 4 exponent, 3 mantissa
//         exponent = (val >> 3) & 0x0F;   // bits 3-6
//         mantissa = val & 0x07;          // bits 0-2
//         if (exponent == 0) {
//             return sign * ldexpf((float)mantissa, -3 - 6); 
//         } 
//         else if (exponent == 0x0F) {
//             return sign * (mantissa ? NAN : INFINITY);
//         }
//         return sign * (1.0f + mantissa / 8.0f) * ldexpf(1.0f, exponent - 7);
//     }
// }

// ===== Host: float -> FP8 (E4M3) 的简易近似转换 =====
static inline uint8_t float_to_fp8_e4m3_noclip(float x) {
    // extern uint8_t float_to_fp8(float val);
    return float_to_fp8(x);
}

// ===== 计算 2D 128x128 分块的 scale =====
// 权重矩阵视为 [K, N] = [REDUCTION_SIZE, OUTPUT_SIZE]
template<typename TFloat> // TFloat = float 或 bfloat16 的可转 float
std::vector<bfloat16> make_scales_2d_blocks(
    const std::vector<TFloat>& w_fp16_like,
    int K, int N,
    int BK=128, int BN=128,
    float e4m3_max=448.0f, float alpha=0.99f)
{
    // K: REDUCTION_SIZE, N: OUTPUT_SIZE
    // BK: block size in K, BN: block size in N
    const int nBK = (K + BK - 1) / BK;
    const int nBN = (N + BN - 1) / BN;
    std::vector<bfloat16> scales(nBK * nBN);

    // for (int br = 0; br < nBK; ++br) {
    //     int k0 = br * BK;
    //     int k1 = std::min(k0 + BK, K);
    //     for (int bc = 0; bc < nBN; ++bc) {
    //         int n0 = bc * BN;
    //         int n1 = std::min(n0 + BN, N);

    //         float amax = 0.f;
    //         for (int k = k0; k < k1; ++k) {
    //             for (int n = n0; n < n1; ++n) {
    //                 float v = static_cast<float>(w_fp16_like[k * N + n]);
    //                 float av = fabsf(v);
    //                 if (av > amax) amax = av;
    //             }
    //         }
    //         // 动态 scale：把块内值映射到 E4M3 可表示范围内（留一点余量）
    //         float s = (amax > 0.f) ? (amax / (alpha * e4m3_max)) : 1.0f;
    //         scales[br * nBN + bc] = bfloat16(s);
    //     }
    // }
    for (int bc = 0; bc < nBN; ++bc) {
        const int n0 = bc * BN;
        const int n1 = std::min(n0 + BN, N);
        for (int br = 0; br < nBK; ++br) {
            const int k0 = br * BK;
            const int k1 = std::min(k0 + BK, K);

            float amax = 0.f;
            // 列主序访问：先按列 n，再按行 k，索引 base = n*K
            for (int n = n0; n < n1; ++n) {
                const int base = n * K; // column-major
                for (int k = k0; k < k1; ++k) {
                    float v = static_cast<float>(w_fp16_like[base + k]);
                    float av = fabsf(v);
                    if (av > amax) amax = av;
                }
            }

            // 把该 128×128 块的最大幅值映射到 E4M3 可表范围（留余量 alpha）
            float s = (amax > 0.f) ? (amax / (alpha * e4m3_max)) : 1.0f;

            // scale 也用列主序：idx = kb + nb * nBK
            scales[br + bc * nBK] = bfloat16(s);
        }
    }
    return scales;
    
}

// ===== 根据 2D block scale 量化到 FP8 =====
template<typename TFloat>
std::vector<uint8_t> quantize_fp8_with_scales_2d(
    const std::vector<TFloat>& w_fp16_like,
    // std::vector<uint8_t>& w_fp8,
    const std::vector<bfloat16>& scales,
    int K, int N, int BK=128, int BN=128)
{
    const int nBK = (K + BK - 1) / BK;
    const int nBN = (N + BN - 1) / BN;

    std::vector<uint8_t> w_fp8(K * N);

    for (int k = 0; k < K; ++k) {
        int br = k / BK;
        for (int n = 0; n < N; ++n) {
            int bc = n / BN;
            float s = static_cast<float>(scales[br * nBN + bc]);
            // 量化：先除以 scale，再 cast 到 FP8
            float x_scaled = static_cast<float>(w_fp16_like[k * N + n]) / (s == 0.f ? 1.f : s);
            w_fp8[k * N + n] = float_to_fp8_e4m3_noclip(x_scaled);
        }
    }
    return w_fp8;
}

// ===== baseline kernel（BF16权重）=====
__global__ void test_linear_kernel_launcher(const void* input, const void* weight,
                                            const void* residual, void* output) {
    extern __shared__ char smem[];
    kernel::linear_kernel<kernel::bfloat16, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>(
        input, weight, residual, output, false);

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[Baseline BF16] rows:\n");
        bfloat16* out = static_cast<bfloat16*>(output);
        for (int n = 0; n < OUTPUT_SIZE * BATCH_SIZE; ++n) {
            printf("%f ", (float)out[n]);
            if ((n + 1) % OUTPUT_SIZE == 0) {
                printf("\n");
            }
            if (n % 12 == 11) { // 每 12 个输出换行
                printf("\n");
            }
        }
        printf("\n");
    }
}

// ===== FP8 权重 kernel（带 scale）=====
__global__ void test_linear_fp8_kernel_launcher_old(const void* input, const void* weight_fp8,
                                                const void* weight_scale, const void* residual, void* output) {
    extern __shared__ char smem[];
    // 你的 kernel 模板可能是：
    // template<typename T, typename FP8, ...>
    // 这里 FP8 用 uint8_t
    // kernel::linear_kernel_fp8_weight<
    //     kernel::bfloat16, uint8_t, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>(
    //         input, weight_fp8, weight_scale, residual, output, true);
    kernel::linear_kernel_fp8_weight1<
        kernel::bfloat16, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>(
            input, weight_fp8, weight_scale, residual, output, false);

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("old [FP8+scale] rows:\n");
        bfloat16* out = static_cast<bfloat16*>(output);
        for (int n = 0; n < OUTPUT_SIZE * BATCH_SIZE; ++n) {
            printf("%f ", (float)out[n]);
            if ((n + 1) % OUTPUT_SIZE == 0) {
                printf("\n");
            }
            if (n % 12 == 11) { // 每 12 个输出换行
                printf("\n");
            }
        }
        // // print the scales for debugging
        // bfloat16* scales = static_cast<bfloat16*>(const_cast<void*>(weight_scale));
        // printf("old Scales:\n");
        // int nBK = (REDUCTION_SIZE + 128 - 1) / 128;
        // int nBN = (OUTPUT_SIZE + 128 - 1) / 128;
        // for (int br = 0; br < nBK; ++br) {
        //     for (int bc = 0; bc < nBN; ++bc) {
        //         printf("%f ", (float)scales[br * nBN + bc]);
        //     }
        // }
        // printf("\n");

    }
}

// ===== FP8 权重 kernel（带 scale）=====
__global__ void test_linear_fp8_kernel_launcher(const void* input, const void* weight_fp8,
                                                const void* weight_scale, const void* residual, void* output) {
    extern __shared__ char smem[];
    // 这里 FP8 用 uint8_t
    // kernel::linear_kernel_fp8_weight<
    //     kernel::bfloat16, uint8_t, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>(
    //         input, weight_fp8, weight_scale, residual, output, true);
    kernel::linear_kernel_fp8_weight<
        kernel::bfloat16, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>(
            input, weight_fp8, weight_scale, residual, output, false);

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[FP8+scale] rows:\n");
        bfloat16* out = static_cast<bfloat16*>(output);
        for (int n = 0; n < OUTPUT_SIZE * BATCH_SIZE; ++n) {
            printf("%f ", (float)out[n]);
            if ((n + 1) % OUTPUT_SIZE == 0) {
                printf("\n");
            }
            if (n % 12 == 11) { // 每 12 个输出换行
                printf("\n");
            }
        }
        // print the scales for debugging
        bfloat16* scales = static_cast<bfloat16*>(const_cast<void*>(weight_scale));
        printf("Scales:\n");
        int nBK = (REDUCTION_SIZE + 128 - 1) / 128;
        int nBN = (OUTPUT_SIZE + 128 - 1) / 128;
        for (int br = 0; br < nBK; ++br) {
            for (int bc = 0; bc < nBN; ++bc) {
                printf("%f ", (float)scales[br * nBN + bc]);
            }
        }
        printf("\n");

    }
}

int main() {
    srand(42);

    cudaFuncSetAttribute(test_linear_kernel_launcher,
    cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARE_MEMORY_SIZE);
    cudaFuncSetAttribute(test_linear_fp8_kernel_launcher,
    cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARE_MEMORY_SIZE);
    cudaFuncSetAttribute(test_linear_fp8_kernel_launcher_old,
    cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARE_MEMORY_SIZE);

    // Host tensors
    std::vector<bfloat16> h_input(BATCH_SIZE * REDUCTION_SIZE);
    std::vector<bfloat16> h_weight_fp16(REDUCTION_SIZE * OUTPUT_SIZE); // KxN
    std::vector<uint8_t> h_weight_fp8(REDUCTION_SIZE * OUTPUT_SIZE); // KxN
    std::vector<bfloat16> h_residual(BATCH_SIZE * OUTPUT_SIZE);
    std::vector<bfloat16> h_output_fp16(BATCH_SIZE * OUTPUT_SIZE);
    std::vector<bfloat16> h_output_fp8(BATCH_SIZE * OUTPUT_SIZE);

    fill_random(h_input,  200.0f);
    // fill_random(h_weight_fp16, 450.0f);
    fill_random_grouped2(h_weight_fp16, REDUCTION_SIZE, OUTPUT_SIZE, 8000.0f, 128, 128);
    fill_random(h_residual, 100000.0f);



    // === 计算 2D block scales（128x128）并据此量化到 FP8 ===
    auto h_scales = make_scales_2d_blocks(h_weight_fp16,
                                          REDUCTION_SIZE, OUTPUT_SIZE,
                                          /*BK=*/128, /*BN=*/128,
                                          /*E4M3_max=*/448.0f, /*alpha=*/0.99f);



    // print scales for debugging
    bool debug2 = false;
    if (debug2) {
        printf("Scales:\n");
        int nBK = (REDUCTION_SIZE + 128 - 1) / 128;
        int nBN = (OUTPUT_SIZE + 128 - 1) / 128;
        for (int br = 0; br < nBK; ++br) {
            for (int bc = 0; bc < nBN; ++bc) {
                printf("%f ", (float)h_scales[br * nBN + bc]);
            }
            printf("\n");
        }
    }

    h_weight_fp8 = quantize_fp8_with_scales_2d(h_weight_fp16, h_scales,
                                                    REDUCTION_SIZE, OUTPUT_SIZE,
                                                    /*BK=*/128, /*BN=*/128);
    // check that there is no NaN in h_weight_fp8
    for (int i = 0; i < h_weight_fp8.size(); ++i) {
        if (h_weight_fp8[i] == 0x7F || h_weight_fp8[i] == 0xFF) {
            printf("Found NaN in h_weight_fp8 at index %d\n", i);
        }
    }
    // print the first 20 weights for debugging
    bool debug1 = true;
    if (debug1) {
        printf("Weight (FP16) sample:\n");
        for (int i = 0; i < 20; ++i) {
            printf("%f ", (float)h_weight_fp16[i]);
        }
        printf("\n");

        printf("Weight (FP8) sample:\n");
        for (int i = 0; i < 20; ++i) {
            // printf("%u ", h_weight_fp8[i]);
            printf("%f ", fp8_to_float(h_weight_fp8[i], false));
        }
        printf("\n");
    }


    // print quantized FP8 weights for debugging
    if (debug2) {
        printf("Quantized FP8 Weights:\n");
        for (int i = 0; i < REDUCTION_SIZE * OUTPUT_SIZE; ++i) {
            // printf("%u ", h_weight_fp8[i]);
            printf("%f ", fp8_to_float(h_weight_fp8[i], false));
            if ((i + 1) % OUTPUT_SIZE == 0) {
                printf("\n");
            }
        }
    }

    // Device tensors
    bfloat16 *d_input=nullptr, *d_weight_fp16=nullptr, *d_residual=nullptr;
    bfloat16 *d_output_fp16=nullptr, *d_output_fp8=nullptr;
    uint8_t  *d_weight_fp8=nullptr;
    bfloat16 *d_scales=nullptr; // scale 存成 BF16（如需 IEEE half，请替换类型）

    CHECK_CUDA(cudaMalloc(&d_input,        h_input.size()        * sizeof(bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_weight_fp16,  h_weight_fp16.size()  * sizeof(bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_weight_fp8,   h_weight_fp8.size()   * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc(&d_residual,     h_residual.size()     * sizeof(bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_output_fp16,  h_output_fp16.size()  * sizeof(bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_output_fp8,   h_output_fp8.size()   * sizeof(bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_scales,       h_scales.size()       * sizeof(bfloat16)));

    CHECK_CUDA(cudaMemcpy(d_input,        h_input.data(),        h_input.size()*sizeof(bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weight_fp16,  h_weight_fp16.data(),  h_weight_fp16.size()*sizeof(bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weight_fp8,   h_weight_fp8.data(),   h_weight_fp8.size()*sizeof(uint8_t),    cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_residual,     h_residual.data(),     h_residual.size()*sizeof(bfloat16),     cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_scales,       h_scales.data(),       h_scales.size()*sizeof(bfloat16),       cudaMemcpyHostToDevice));

    // print the input, weight and residual for debugging
    bool debug = false; // 仅在调试时打印
    if (debug) {// 仅在调试时打印
        printf("Input:\n");
        for (int i = 0; i < BATCH_SIZE * REDUCTION_SIZE; ++i) {
            printf("%f ", (float)h_input[i]);
            if ((i + 1) % REDUCTION_SIZE == 0) {
                printf("\n");
            }
        }
        printf("Weight (FP16):\n");
        for (int i = 0; i < REDUCTION_SIZE * OUTPUT_SIZE; ++i) {
            printf("%f ", (float)h_weight_fp16[i]);
            if ((i + 1) % OUTPUT_SIZE == 0) {
                printf("\n");
            }
        }
        printf("Weight (FP8):\n");
        for (int i = 0; i < REDUCTION_SIZE * OUTPUT_SIZE; ++i) {
            printf("%u ", h_weight_fp8[i]);
            if ((i + 1) % OUTPUT_SIZE == 0) {
                printf("\n");
            }
        }
        printf("Residual:\n");
        for (int i = 0; i < BATCH_SIZE * OUTPUT_SIZE; ++i) {
            printf("%f ", (float)h_residual[i]);
            if ((i + 1) % OUTPUT_SIZE == 0) {
                printf("\n");
            }
        }
        printf("Scales:\n");
        for (int i = 0; i < h_scales.size(); ++i) {
            printf("%f ", (float)h_scales[i]);
            if ((i + 1) % ((REDUCTION_SIZE + 127) / 128) == 0) {
                printf("\n");
            }
        }
        printf("\n");
    }   

    // 计算需要的 shared memory（这里仍然用固定值，按需调整）
    // size_t smem_size = 30000;

    // baseline
    printf("Max shared memory size: %zu bytes\n", MAX_SHARE_MEMORY_SIZE);
    test_linear_kernel_launcher<<<1, 128, MAX_SHARE_MEMORY_SIZE>>>(d_input, d_weight_fp16, d_residual, d_output_fp16);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    // fp8 + scale
    test_linear_fp8_kernel_launcher<<<1, 128, MAX_SHARE_MEMORY_SIZE>>>(d_input, d_weight_fp8, d_scales, d_residual, d_output_fp8);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    // fp8 + scale
    test_linear_fp8_kernel_launcher_old<<<1, 128, MAX_SHARE_MEMORY_SIZE>>>(d_input, d_weight_fp8, d_scales, d_residual, d_output_fp8);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    // 拷回结果
    CHECK_CUDA(cudaMemcpy(h_output_fp16.data(), d_output_fp16, h_output_fp16.size()*sizeof(bfloat16), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output_fp8.data(),  d_output_fp8,  h_output_fp8.size()*sizeof(bfloat16),  cudaMemcpyDeviceToHost));

    // 误差评估
    float max_err = 0.0f, mae=0.f, mre=0.f, mean_rel_err=0.f;
    for (size_t i = 0; i < h_output_fp16.size(); ++i) {
        float ref = (float)h_output_fp16[i];
        float tst = (float)h_output_fp8[i];
        float e = fabsf(ref - tst);
        mre = fmaxf(mre, e / (fabsf(ref) + 10000.0f)); // max relative error
        mean_rel_err += e / (fabsf(ref) + 10000.0f); // mean relative error
        max_err = fmaxf(max_err, e);
        mae += e;
    }
    mean_rel_err /= (float)h_output_fp16.size();
    mae /= (float)h_output_fp16.size();
    std::cout << ", Max rel error: " << mre << ", Mean rel error: " << mean_rel_err << std::endl;

    // const int warmup = 10;
    // const int iters  = 200;

    // auto launch_baseline = [&](){
    //     test_linear_kernel_launcher<<<1, 128, MAX_SHARE_MEMORY_SIZE>>>(d_input, d_weight_fp16, d_residual, d_output_fp16);
    // };
    // auto launch_fp8 = [&](){
    //     test_linear_fp8_kernel_launcher<<<1, 128, MAX_SHARE_MEMORY_SIZE>>>(d_input, d_weight_fp8, d_scales, d_residual, d_output_fp8);
    // };

    // // Time both
    // BenchResult base_ms = time_kernel(warmup, iters, launch_baseline);
    // BenchResult fp8_ms  = time_kernel(warmup, iters, launch_fp8);

    // // Derived metrics
    // const int M = BATCH_SIZE, N = OUTPUT_SIZE, K = REDUCTION_SIZE;
    // const double flops      = gemm_flops(M, N, K); // per run
    // const double bytes_bf16 = bytes_moved_baseline_bf16(M, N, K);
    // const double bytes_fp8  = bytes_moved_fp8_scale(M, N, K, 128, 128);

    // auto to_tflops = [&](float ms){ return (flops / (ms * 1e-3)) / 1e12; };
    // auto to_gbps   = [&](double bytes, float ms){ return (bytes / (ms * 1e-3)) / 1e9; };

    // printf("\n==== Performance (M=%d, N=%d, K=%d; %d warmup, %d iters) ====\n", M, N, K, warmup, iters);
    // printf("Baseline BF16: avg %.3f ms  (std %.3f)\n", base_ms.avg_ms, base_ms.std_ms);
    // printf("  ~Throughput: %.2f GB/s  |  ~Perf: %.3f TFLOP/s\n",
    //        to_gbps(bytes_bf16, base_ms.avg_ms), to_tflops(base_ms.avg_ms));

    // printf("FP8 + scale : avg %.3f ms  (std %.3f)\n", fp8_ms.avg_ms, fp8_ms.std_ms);
    // printf("  ~Throughput: %.2f GB/s  |  ~Perf: %.3f TFLOP/s\n",
    //        to_gbps(bytes_fp8, fp8_ms.avg_ms), to_tflops(fp8_ms.avg_ms));


    // 你可以根据模型容忍度改阈值
    if (max_err < 2.0f) {
        std::cout << "PASS: FP8(+scale) kernel reasonably matches baseline." << std::endl;
    } else {
        std::cout << "FAIL: Error too high!" << std::endl;
    }

    // 释放
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_weight_fp16));
    CHECK_CUDA(cudaFree(d_weight_fp8));
    CHECK_CUDA(cudaFree(d_residual));
    CHECK_CUDA(cudaFree(d_output_fp16));
    CHECK_CUDA(cudaFree(d_output_fp8));
    CHECK_CUDA(cudaFree(d_scales));

    return 0;
}
