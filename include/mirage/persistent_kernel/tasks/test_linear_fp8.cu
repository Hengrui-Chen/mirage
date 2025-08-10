#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <assert.h>
#include "linear.cuh"  // 假设包含 linear_kernel 和 linear_kernel_fp8_weight
#include <stdio.h>

using bfloat16 = kernel::bfloat16;

// 参数定义
// constexpr int BATCH_SIZE = 16;
// constexpr int OUTPUT_SIZE = 64;
// constexpr int REDUCTION_SIZE = 128;

// constexpr int BATCH_SIZE = 8;
// constexpr int OUTPUT_SIZE = 16;
// constexpr int REDUCTION_SIZE = 32;

// constexpr int BATCH_SIZE = 8;
// constexpr int OUTPUT_SIZE = 16;
// constexpr int REDUCTION_SIZE = 128 * 2;

constexpr int BATCH_SIZE = 1;
constexpr int OUTPUT_SIZE = 64;
constexpr int REDUCTION_SIZE = 512;

// extern int MAX_SHARE_MEMORY_SIZE; // 在其他地方定义的宏，表示最大共享内存大小

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
constexpr int MAX_SHARE_MEMORY_SIZE = 224 * 1024;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
constexpr int MAX_SHARE_MEMORY_SIZE = 96 * 1024;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
constexpr int MAX_SHARE_MEMORY_SIZE = 160 * 1024;
#else
constexpr int MAX_SHARE_MEMORY_SIZE = 96 * 1024;
#endif

template<typename T>
void fill_random(std::vector<T>& vec, float scale=100.0f) {
    for (auto& v : vec) {
        v = static_cast<T>((static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2 * scale);
        // v = static_cast<T>(0); // for debug
    }
    // vec[0] = static_cast<T>(0); // for debug
}

// 包装kernel调用linear_kernel (FP16 baseline)
__global__ void test_linear_kernel_launcher(const void* input, const void* weight, 
                                            const void* residual, void* output) {
    // // print the input elements
    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("Input elements:\n");
    //     bfloat16* input_ptr = static_cast<bfloat16*>(const_cast<void*>(input));
    //     for (int i = 0; i < BATCH_SIZE * REDUCTION_SIZE; i++) {
    //         printf("%f ", static_cast<float>(input_ptr[i]));
    //         // newline for each input row
    //         if ((i+1) % REDUCTION_SIZE == 0) {
    //             printf("\n");
    //         }
    //     }
    //     printf("\n");
    // }
    extern __shared__ char smem[];
    kernel::linear_kernel<kernel::bfloat16, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>(
        input, weight, residual, output, false);

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Output elements:\n");
        bfloat16* output_ptr = static_cast<bfloat16*>(output);
        for (int i = 0;  i < BATCH_SIZE * OUTPUT_SIZE; i++) {
            // i < 20 &&
            printf("%f ", static_cast<float>(output_ptr[i]));
            // newline for each output row
            if ((i+1) % OUTPUT_SIZE == 0) {
                printf("\n");
            }
        }
        printf("\n");
    }
}

// 包装kernel调用linear_kernel_fp8_weight (FP8 weight)
__global__ void test_linear_fp8_kernel_launcher(const void* input, const void* weight_fp8,
                                                const void* scale, const void* residual, void* output) {
    extern __shared__ char smem[];
    kernel::linear_kernel_fp8_weight<kernel::bfloat16, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>(
        input, weight_fp8, scale, residual, output, false);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Output first elements (FP8):\n");
        bfloat16* output_ptr = static_cast<bfloat16*>(output);
        for (int i = 0;  i < BATCH_SIZE * OUTPUT_SIZE; i++) {
            printf("%f ", static_cast<float>(output_ptr[i]));
            // newline for each output row
            if ((i+1) % OUTPUT_SIZE == 0) {
                printf("\n");
            }
        }
        printf("\n");
    }
}

// host 上的 FP16 -> FP8 (E4M3) 量化
uint8_t float_to_fp8(float val) {
    int sign = val < 0 ? 1 : 0;
    val = fabsf(val);

    if (val == 0) return 0;

    int exp;
    float mant = frexpf(val, &exp); // val = mant * 2^(exp)
    exp += 6; // 对应 E4M3 偏移量

    if (exp <= 0) { // 次正规数
        int m = (int)(mant * (1 << (exp + 2))); 
        return (sign << 7) | (0 << 3) | (m & 0x7);
    } else if (exp >= 0xF) { // 溢出
        return (sign << 7) | (0xF << 3); // Inf
    } else { 
        int m = (int)((mant - 0.5f) * (1 << 4)); 
        return (sign << 7) | ((exp & 0xF) << 3) | (m & 0x7);
    }
}

float fp8_to_float(uint8_t val, bool e5m2 = false) {
    // Sign extraction
    int sign = (val & 0x80) ? -1 : 1; // MSB is sign bit
    int exponent, mantissa;

    if (e5m2) {
        // FP8 E5M2: 1 sign, 5 exponent, 2 mantissa
        exponent = (val >> 2) & 0x1F;   // bits 2-6
        mantissa = val & 0x03;          // bits 0-1
        if (exponent == 0) {
            // Subnormal
            return sign * ldexpf((float)mantissa, -2 - 14); 
        } else if (exponent == 0x1F) {
            return sign * (mantissa ? NAN : INFINITY); 
        }
        return sign * (1.0f + mantissa / 4.0f) * ldexpf(1.0f, exponent - 15);
    } 
    else {
        // FP8 E4M3: 1 sign, 4 exponent, 3 mantissa
        exponent = (val >> 3) & 0x0F;   // bits 3-6
        mantissa = val & 0x07;          // bits 0-2
        if (exponent == 0) {
            return sign * ldexpf((float)mantissa, -3 - 6); 
        } else if (exponent == 0x0F) {
            return sign * (mantissa ? NAN : INFINITY);
        }
        return sign * (1.0f + mantissa / 8.0f) * ldexpf(1.0f, exponent - 7);
    }
}

int main() {
    srand(42);

    cudaFuncSetAttribute(test_linear_kernel_launcher,
    cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARE_MEMORY_SIZE);
    cudaFuncSetAttribute(test_linear_fp8_kernel_launcher,
    cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARE_MEMORY_SIZE);

    // Host memory
    std::vector<bfloat16> h_input(BATCH_SIZE * REDUCTION_SIZE);
    std::vector<bfloat16> h_weight_fp16(REDUCTION_SIZE * OUTPUT_SIZE);
    std::vector<uint8_t> h_weight_fp8(REDUCTION_SIZE * OUTPUT_SIZE);
    std::vector<bfloat16> h_residual(BATCH_SIZE * OUTPUT_SIZE);
    std::vector<bfloat16> h_output_fp16(BATCH_SIZE * OUTPUT_SIZE);
    std::vector<bfloat16> h_output_fp8(BATCH_SIZE * OUTPUT_SIZE);

    // 填充数据
    fill_random(h_input, 100.0f);
    fill_random(h_weight_fp16, 100.0f);
    fill_random(h_residual, 10000.0f);

    // 量化权重到 FP8
    for (int i = 0; i < REDUCTION_SIZE * OUTPUT_SIZE; ++i) {
        float w = (float)h_weight_fp16[i];
        h_weight_fp8[i] = float_to_fp8(w);
    }

    bfloat16 scale = static_cast<bfloat16>(1.0f); // 假设 scale 为 1.0f

    // 重新载入 bfloat16 数据
    for (int i = 0; i < REDUCTION_SIZE * OUTPUT_SIZE; ++i) {
        float t = fp8_to_float(h_weight_fp8[i], false); // E4M3
        h_weight_fp16[i] = static_cast<bfloat16>(t) * scale; // 重新缩放
    }

    // 设备内存分配
    bfloat16 *d_input, *d_weight_fp16, *d_residual, *d_output_fp16, *d_output_fp8;
    uint8_t *d_weight_fp8;

    cudaMalloc(&d_input, h_input.size() * sizeof(bfloat16));
    cudaMalloc(&d_weight_fp16, h_weight_fp16.size() * sizeof(bfloat16));
    cudaMalloc(&d_weight_fp8, h_weight_fp8.size() * sizeof(uint8_t));
    cudaMalloc(&d_residual, h_residual.size() * sizeof(bfloat16));
    cudaMalloc(&d_output_fp16, h_output_fp16.size() * sizeof(bfloat16));
    cudaMalloc(&d_output_fp8, h_output_fp8.size() * sizeof(bfloat16));

    cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_fp16, h_weight_fp16.data(), h_weight_fp16.size() * sizeof(bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_fp8, h_weight_fp8.data(), h_weight_fp8.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_residual, h_residual.data(), h_residual.size() * sizeof(bfloat16), cudaMemcpyHostToDevice);

    size_t smem_size;  // shared memory size,根据需求调整
    smem_size = 30000;
    smem_size = 40960;
    smem_size = sizeof(bfloat16) * 3 * 128 * 128 ;
    printf("Shared memory size: %zu bytes\n", smem_size);

    // print the residual for debugging
    if (true) { // 仅在调试时打印 --- IGNORE ---
        printf("Residual:\n");
        for (int i = 0; i < BATCH_SIZE * OUTPUT_SIZE; ++i) {
            printf("%f ", (float)h_residual[i]);
        }
        printf("\n");
    }


    // baseline kernel launch
    test_linear_kernel_launcher<<<1, 128, MAX_SHARE_MEMORY_SIZE>>>(
        d_input, d_weight_fp16, d_residual, d_output_fp16);
    // fflush(stdout);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error1: %s\n", cudaGetErrorString(err));
    }


    // fp8 kernel launch
    test_linear_fp8_kernel_launcher<<<1, 128, MAX_SHARE_MEMORY_SIZE>>>(
        d_input, d_weight_fp8, nullptr, d_residual, d_output_fp8);
    // fflush(stdout);

    cudaDeviceSynchronize();
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        printf("CUDA Error2: %s\n", cudaGetErrorString(err2));
        fprintf(stderr, "CUDA Error at %s:%d \n", __FILE__, __LINE__); 
    }

    // 拷贝结果回 host
    cudaMemcpy(h_output_fp16.data(), d_output_fp16, h_output_fp16.size() * sizeof(bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_fp8.data(), d_output_fp8, h_output_fp8.size() * sizeof(bfloat16), cudaMemcpyDeviceToHost);

    // 误差比较
    float max_err = 0.0f;
    for (size_t i = 0; i < h_output_fp16.size(); ++i) {
        // potential errors! you can't just convert fp8 (uint8_t) back to float and compare
        float ref = (float)h_output_fp16[i];
        float test = (float)h_output_fp8[i];
        // printf("Index %zu: FP16=%.6f, FP8=%.6f\n", i, ref, test);
        float err = fabs(ref - test);
        max_err = fmax(max_err, err);
    }

    std::cout << "Max absolute error: " << max_err << std::endl;
    if (max_err < 1e-2) {
        std::cout << "PASS: FP8 kernel matches FP16 baseline within tolerance." << std::endl;
    } else {
        std::cout << "FAIL: Error too high!" << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_weight_fp16);
    cudaFree(d_weight_fp8);
    cudaFree(d_residual);
    cudaFree(d_output_fp16);
    cudaFree(d_output_fp8);

    return 0;
}
