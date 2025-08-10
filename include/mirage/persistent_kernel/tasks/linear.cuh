
/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "common.h"
#include "copy_sm80.cuh"
#include "dmem_layout.cuh"
#include "element_binary.cuh"
#include "element_unary.cuh"
#include "mma.cuh"
#include "reduction.cuh"
#include "smem_layout.cuh"
#include "utils.cuh"
#include <stdio.h>
#include <cstdio>

namespace kernel {

using bfloat16 = type::bfloat16_t;

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int O_STRIDE = OUTPUT_SIZE,
          int K_PIPE_MAX = 3>
__device__ __forceinline__ void linear_kernel(void const *input_ptr,
                                              void const *weight_ptr,
                                              void const *residual_ptr,
                                              void *output_ptr,
                                              bool residual = true) {
    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("Reached point 45\n");
    // }
  constexpr int CHUNK_SIZE = 16 / sizeof(T);
  constexpr int OUTPUT_ATOM_SIZE = OUTPUT_SIZE <= 128 ? OUTPUT_SIZE : 128;
  constexpr int NUM_OUTPUT_ATOMS = OUTPUT_SIZE / OUTPUT_ATOM_SIZE;
  constexpr int TILE_SIZE = 128;
  constexpr int FORLOOP_RANGE = REDUCTION_SIZE / TILE_SIZE;

  constexpr int NUM_CHUNKS_A = BATCH_SIZE * TILE_SIZE / CHUNK_SIZE;
  constexpr int NUM_CHUNKS_B = TILE_SIZE * OUTPUT_ATOM_SIZE / CHUNK_SIZE;
  constexpr int NUM_CHUNKS_C = BATCH_SIZE * OUTPUT_ATOM_SIZE / CHUNK_SIZE;

  constexpr int CHUNKS_PER_ROW_A = TILE_SIZE / CHUNK_SIZE;
  constexpr int CHUNKS_PER_COL_B = TILE_SIZE / CHUNK_SIZE;
  constexpr int CHUNKS_PER_ROW_C = OUTPUT_ATOM_SIZE / CHUNK_SIZE;

  constexpr int log2_CHUNK_SIZE = log2_constexpr(CHUNK_SIZE);
  constexpr int log2_CHUNKS_PER_ROW_A = log2_constexpr(CHUNKS_PER_ROW_A);
  constexpr int log2_CHUNKS_PER_COL_B = log2_constexpr(CHUNKS_PER_COL_B);
  constexpr int log2_CHUNKS_PER_ROW_C = log2_constexpr(CHUNKS_PER_ROW_C);

  // using SM80_16x8x16_F16F16F16F16_TNX2 = 16X16X16
  constexpr int NUM_WARPS_N =
      OUTPUT_ATOM_SIZE / 16 <= 4 ? OUTPUT_ATOM_SIZE / 16 : 4;
  constexpr int NUM_WARPS_K = 4 / NUM_WARPS_N;

  constexpr int NUM_ITERS_M = 1;
  constexpr int NUM_ITERS_N = OUTPUT_ATOM_SIZE / NUM_WARPS_N / 16;
  constexpr int NUM_ITERS_K = TILE_SIZE / NUM_WARPS_K / 16;

  constexpr int log2_NUM_WARPS_N = log2_constexpr(NUM_WARPS_N);
  constexpr int log2_NUM_ITERS_K = log2_constexpr(NUM_ITERS_K);

  int warp_idx = warp_id();
  int warp_row = warp_idx >> log2_NUM_WARPS_N;
  int warp_col = warp_idx & (NUM_WARPS_N - 1);
  int lane_idx = lane_id();

  T const *__restrict__ d_input = static_cast<T const *>(input_ptr);
  T const *__restrict__ d_weight = static_cast<T const *>(weight_ptr);
  T const *__restrict__ d_residual =
      residual ? static_cast<T const *>(residual_ptr) : nullptr;
  T *__restrict__ d_output = static_cast<T *>(output_ptr);

  using InputDmem = dmem_row_const<T, BATCH_SIZE, TILE_SIZE, REDUCTION_SIZE>;
  using WeightDmem =
      dmem_col_const<T, TILE_SIZE, OUTPUT_ATOM_SIZE, REDUCTION_SIZE>;
  using ResidualDmem =
      dmem_row_const<T, BATCH_SIZE, OUTPUT_ATOM_SIZE, O_STRIDE>;
  using OutputDmem = dmem_row<T, BATCH_SIZE, OUTPUT_ATOM_SIZE, O_STRIDE>;

  InputDmem input_dmem(d_input);
  WeightDmem weight_dmem(d_weight);
  ResidualDmem residual_dmem(d_residual);
  OutputDmem output_dmem(d_output);

  extern __shared__ char smem[];

  // STensors' offsets
  constexpr size_t ZERO_BUFFER_OFFSET = 0;
  // sizeof(T) * 8

  constexpr size_t SHARED_INPUT_BUFFER_OFFSET =
      ZERO_BUFFER_OFFSET + sizeof(T) * 8;
  // sizeof(T) * K_PIPE_MAX * BATCH_SIZE * TILE_SIZE

  constexpr size_t SHARED_WEIGHT_BUFFER_OFFSET =
      SHARED_INPUT_BUFFER_OFFSET +
      sizeof(T) * K_PIPE_MAX * BATCH_SIZE * TILE_SIZE;
  // sizeof(T) * K_PIPE_MAX * TILE_SIZE * OUTPUT_ATOM_SIZE

  constexpr size_t SHARED_RESIDUAL_OFFSET =
      SHARED_WEIGHT_BUFFER_OFFSET +
      sizeof(T) * K_PIPE_MAX * TILE_SIZE * OUTPUT_ATOM_SIZE;
  // sizeof(T) * BATCH_SIZE * OUTPUT_ATOM_SIZE

  constexpr size_t MM_INTERMEDIATE_OFFSET =
      SHARED_RESIDUAL_OFFSET + sizeof(T) * BATCH_SIZE * OUTPUT_ATOM_SIZE;
  // sizeof(T) * NUM_WARPS_K * BATCH_SIZE * OUTPUT_ATOM_SIZE

  constexpr size_t SHARED_OUTPUT_OFFSET =
      MM_INTERMEDIATE_OFFSET +
      sizeof(T) * NUM_WARPS_K * BATCH_SIZE * OUTPUT_ATOM_SIZE;
  // sizeof(T) * BATCH_SIZE * OUTPUT_ATOM_SIZE

  // zero buffer
  T *zero_buf = (T *)(smem + ZERO_BUFFER_OFFSET);
  vec_zero_t<T, 8>::fill_zero(zero_buf);

  // copy
  T *shared_input_buffer = (T *)(smem + SHARED_INPUT_BUFFER_OFFSET);
  T *shared_weight_buffer = (T *)(smem + SHARED_WEIGHT_BUFFER_OFFSET);

  // residual
  T *shared_residual =
      residual ? (T *)(smem + SHARED_RESIDUAL_OFFSET) : nullptr;

  // intermediate
  T *mm_intermediate = (T *)(smem + MM_INTERMEDIATE_OFFSET);

  // output
  T *shared_output = (T *)(smem + SHARED_OUTPUT_OFFSET);

  // define the swizzle mode
  using ZeroBufferSmem = smem_row<T, 0, 0, 0, 1, 8, 8>;
  using InputSmem = smem_row<T, 0, 0, 0, BATCH_SIZE, TILE_SIZE, TILE_SIZE>;
  using InputBufferSmem =
      smem_row<T, 0, 0, 0, K_PIPE_MAX * BATCH_SIZE, TILE_SIZE, TILE_SIZE>;
  using WeightSmem =
      smem_col<T, 3, 3, 3, TILE_SIZE, OUTPUT_ATOM_SIZE, TILE_SIZE>;
  using WeightBufferSmem =
      smem_col<T, 3, 3, 3, TILE_SIZE, K_PIPE_MAX * OUTPUT_ATOM_SIZE, TILE_SIZE>;
  using OutputSmem =
      smem_row<T, 0, 0, 0, BATCH_SIZE, OUTPUT_ATOM_SIZE, OUTPUT_ATOM_SIZE>;
  using MatMulIntermediateSmem = smem_row<T,
                                          0,
                                          0,
                                          0,
                                          NUM_WARPS_K * BATCH_SIZE,
                                          OUTPUT_ATOM_SIZE,
                                          OUTPUT_ATOM_SIZE>;

  ZeroBufferSmem zero_buffer(zero_buf);

  InputSmem input_smem(shared_input_buffer);
  WeightSmem weight_smem(shared_weight_buffer);

  OutputSmem residual_smem(shared_residual);

  MatMulIntermediateSmem mm_intermediate_smem(mm_intermediate);

  OutputSmem output_smem(shared_output);

//   print first 20 elements starting from input_ptr, save it to local file
    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("Input first 20 elements:\n");
    //     for (int i = 0; i < 20 && i < BATCH_SIZE * TILE_SIZE; i++) {
    //     printf("%f ", static_cast<float>(d_input[i]));
    //     }
    //     printf("\n");
    // }

// if (threadIdx.x == 0 && blockIdx.x == 0) {
//     printf("Reached point A\n");
// }
// // printf("Reached point 191\n");
  for (int output_atom_idx = 0; output_atom_idx < NUM_OUTPUT_ATOMS;
       output_atom_idx++,
           d_weight += OUTPUT_ATOM_SIZE * REDUCTION_SIZE,
           d_residual = residual ? d_residual + OUTPUT_ATOM_SIZE : nullptr,
           d_output += OUTPUT_ATOM_SIZE) {
    weight_dmem.set_ptr(d_weight);
    residual_dmem.set_ptr(d_residual);
    output_dmem.set_ptr(d_output);

    InputBufferSmem input_buffer_smem(shared_input_buffer);
    WeightBufferSmem weight_buffer_smem(shared_weight_buffer);

    if (residual) {
#pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_C; i += NUM_THREADS) {
        int row = i >> log2_CHUNKS_PER_ROW_C;
        int col = (i & (CHUNKS_PER_ROW_C - 1)) << log2_CHUNK_SIZE;
        load_smem(residual_smem(row, col), residual_dmem(row, col));
      }
    }

#pragma unroll
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; k_pipe++) {
#pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_A; i += NUM_THREADS) {
        int src_row = i >> log2_CHUNKS_PER_ROW_A;
        int dst_row = src_row + ((k_pipe + 1) * BATCH_SIZE);
        int dst_col = (i & (CHUNKS_PER_ROW_A - 1)) << log2_CHUNK_SIZE;
        int src_col = dst_col + (k_pipe << log2_constexpr(TILE_SIZE));
        load_smem(input_buffer_smem(dst_row, dst_col),
                  input_dmem(src_row, src_col));
      }
#pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_B; i += NUM_THREADS) {
        int dst_row = (i & (CHUNKS_PER_COL_B - 1)) << log2_CHUNK_SIZE;
        int src_row = dst_row + (k_pipe << log2_constexpr(TILE_SIZE));
        int src_col = i >> log2_CHUNKS_PER_COL_B;
        int dst_col =
            src_col + ((k_pipe + 1) << log2_constexpr(OUTPUT_ATOM_SIZE));
        load_smem(weight_buffer_smem(dst_row, dst_col),
                  weight_dmem(src_row, src_col));
      }
      cp_async_fence();
    }

    // accumulator
    alignas(16) float s_frag[NUM_ITERS_M][NUM_ITERS_N][8];
    for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
#pragma unroll
      for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
        clear_8_floats(s_frag[m][n]);
      }
    }

    // please check weather uncomment these two lines will affect finally result
    // cp_async_wait<0>();
    // __syncthreads();


    for (int for_idx = 0; for_idx < FORLOOP_RANGE; for_idx++) {
      // copy
      if (for_idx + K_PIPE_MAX - 1 < FORLOOP_RANGE) {
#pragma unroll
        for (int i = threadIdx.x; i < NUM_CHUNKS_A; i += NUM_THREADS) {
          int row = i >> log2_CHUNKS_PER_ROW_A;
          int dst_col = (i & (CHUNKS_PER_ROW_A - 1)) << log2_CHUNK_SIZE;
          int src_col = dst_col + ((for_idx + K_PIPE_MAX - 1)
                                   << log2_constexpr(TILE_SIZE));
          load_smem(input_buffer_smem(row, dst_col), input_dmem(row, src_col));
        }
#pragma unroll
        for (int i = threadIdx.x; i < NUM_CHUNKS_B; i += NUM_THREADS) {
          int dst_row = (i & (CHUNKS_PER_COL_B - 1)) << log2_CHUNK_SIZE;
          int src_row = dst_row + ((for_idx + K_PIPE_MAX - 1)
                                   << log2_constexpr(TILE_SIZE));
          int col = i >> log2_CHUNKS_PER_COL_B;
          load_smem(weight_buffer_smem(dst_row, col),
                    weight_dmem(src_row, col));
        }
        cp_async_fence();
        cp_async_wait<K_PIPE_MAX - 1>();
      } else if (for_idx + K_PIPE_MAX - 1 == FORLOOP_RANGE) {
        cp_async_wait<0>();
      }

      // rotate the buffers
      input_buffer_smem.set_ptr(shared_input_buffer +
                                BATCH_SIZE * TILE_SIZE *
                                    ((for_idx + 1) % K_PIPE_MAX));
      input_smem.set_ptr(shared_input_buffer +
                         BATCH_SIZE * TILE_SIZE * ((for_idx + 1) % K_PIPE_MAX));
      weight_buffer_smem.set_ptr(shared_weight_buffer +
                                 TILE_SIZE * OUTPUT_ATOM_SIZE *
                                     ((for_idx + 1) % K_PIPE_MAX));
      weight_smem.set_ptr(shared_weight_buffer +
                          TILE_SIZE * OUTPUT_ATOM_SIZE *
                              ((for_idx + 1) % K_PIPE_MAX));
      __syncthreads();
        // if (threadIdx.x == 0 && blockIdx.x == 0)
        //     printf("Debug: tid = %d\n", threadIdx.x);
        // __syncthreads();


      uint32_t a_frag[4], b_frag[4];
      for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
        int m_row = (lane_idx & 0xF);
        bool is_valid = (m_row < BATCH_SIZE);
#pragma unroll
        for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
          int n_col = (n << (4 + log2_NUM_WARPS_N)) + (warp_col << 4) +
                      ((lane_idx >> 4) << 3) + (lane_idx & 0x7);
#pragma unroll
          for (uint32_t k = 0; k < NUM_ITERS_K; k++) {
            int m_col = (warp_row << (4 + log2_NUM_ITERS_K)) + (k << 4) +
                        ((lane_idx >> 4) << 3);
            int n_row = (warp_row << (4 + log2_NUM_ITERS_K)) + (k << 4) +
                        (((lane_idx & 0xF) >> 3) << 3);
            T *src_ptr =
                is_valid ? input_smem(m_row, m_col) : zero_buffer(0, 0);
            ldsm(src_ptr, a_frag);
            ldsm(weight_smem(n_row, n_col), b_frag);
            mma_m16n16k16_bf16bf16bf32(
                s_frag[m][n], a_frag, b_frag, s_frag[m][n]);
            // printf("M: %d, N: %d, K: %d, a_frag: %u, b_frag: %u\n",
                //    m_row, n_col, k, a_frag[0], b_frag[0]);
            

          }
        }
      }
      __syncthreads();
    }

    // write back to shared memory
    for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
#pragma unroll
      for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
#pragma unroll
        for (uint32_t i = 0; i < 4; i++) {
          int row_in_warp = (lane_idx >> 2) + ((i & 0x1) << 3);
          if (row_in_warp < BATCH_SIZE) {
            int col = (n << (4 + log2_NUM_WARPS_N)) + (warp_col << 4) +
                      ((lane_idx & 0x3) << 1) + ((i >> 1) << 3);
            mm_intermediate_smem.at(warp_row + row_in_warp, col) =
                bfloat16(s_frag[m][n][(i << 1)]);
            mm_intermediate_smem.at(warp_row + row_in_warp, col + 1) =
                bfloat16(s_frag[m][n][(i << 1) | 0x1]);
          }
        // check if the element is not equal to zero (raise warning if it is)
        

        }
      }
    }
    // printf("Reached point 341\n");
    __syncthreads();

    if (NUM_WARPS_K > 1) {
      reduction_sum_row<decltype(output_smem), decltype(mm_intermediate_smem)>(
          output_smem, mm_intermediate_smem);
      __syncthreads();
    }

#pragma unroll
    for (int row = 0; row < BATCH_SIZE; row++) {
#pragma unroll
      for (int i = threadIdx.x; i < OUTPUT_ATOM_SIZE; i += NUM_THREADS) {
        T val = NUM_WARPS_K > 1 ? output_smem.at(row, i)
                                : mm_intermediate_smem.at(row, i);
        output_dmem.at(row, i) =
            residual ? val + residual_smem.at(row, i) : val;
      }
    }
    if (output_atom_idx + 1 < NUM_OUTPUT_ATOMS) {
      __syncthreads();
    }
    // printf("Reached point 363\n");
  }
}


__global__ void hello_from_gpu() {
    printf("Hello from thread %d, block %d\n", threadIdx.x, blockIdx.x);
}




__device__ __forceinline__ bfloat16 fp8_to_fp16(uint8_t fp8_val) {
    // E4M3 FP8 format to FP16 (half precision)
    int sign = (fp8_val & 0x80) ? -1 : 1;
    int exp  = (fp8_val & 0x78) >> 3;
    int mant = (fp8_val & 0x07);

    if (exp == 0) {
        float val = sign * mant * powf(2, -6);  // subnormal
        return bfloat16(val);
    } else if (exp == 0xF) {
        return bfloat16(sign * INFINITY);   // Inf/NaN
    } else {
        float val = sign * (1.0f + mant / 8.0f) * powf(2, exp - 7);
        return bfloat16(val);
    }
}

__global__ void convert_fp8_to_fp16_kernel(
    const uint8_t* __restrict__ src_fp8,
    bfloat16* __restrict__ dst_fp16,
    int rows, int cols, int stride_fp8, int stride_fp16) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        // Calculate source and destination index considering stride
        int src_idx = row  + col * stride_fp8;
        int dst_idx = row  + col * stride_fp16;

        dst_fp16[dst_idx] = fp8_to_fp16(src_fp8[src_idx]);
    }
}


void convert_fp8_to_fp16(const uint8_t* weight_fp8,
                         bfloat16* weight_fp16,
                         int rows, int cols,
                         int stride_fp8, int stride_fp16) 
{
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);

    convert_fp8_to_fp16_kernel<<<grid, block>>>(
        weight_fp8, weight_fp16, rows, cols, stride_fp8, stride_fp16);
    cudaDeviceSynchronize();
}


// load 128 bytes values from global to shared memory async
template <typename FP8>
__device__ __forceinline__ void load_smem_raw(FP8 *smem_ptr, FP8 const *gmem_ptr) {
#ifdef CP_ASYNC_SM80_ENABLED
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(
                   smem_int_ptr),
               "l"(gmem_ptr),
               "n"(16),
               "r"(16));
#endif
}


// // Assert that a pointer is aligned to a specific boundary
__device__ __forceinline__ void assert_aligned(const void* ptr, size_t alignment, const char* name) {
    assert((reinterpret_cast<uintptr_t>(ptr) % alignment == 0) && "Pointer alignment check failed!");
}



// __device__ __forceinline__ half fp8_to_fp16(uint8_t val, half scale) {
//     // Example: FP8 (E4M3 or E5M2) decoding logic (simplified)
//     float f = decode_fp8_to_fp32(val);   // hardware/inline decode
//     return __float2half(f) * scale;
// }

__device__ __forceinline__ float fp8_to_float(uint8_t val, bool e5m2 = false) {
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

// template <typename FP8>
// __device__ __forceinline__ void load_smem_raw(FP8 *dst, FP8 const *src) {
// #pragma unroll
//     for (int i = 0; i < 16 / sizeof(FP8); i++) { // 16B (128-bit) aligned loads per thread
//         dst[i] = src[i];
//     }
// }

// Load 128B FP8 data from global to shared memory asynchronously
// template <typename FP8>
// __device__ __forceinline__ void load_smem_raw(FP8 *smem_ptr, FP8 const *gmem_ptr) {
//     // Alignment checks
//     assert((reinterpret_cast<uintptr_t>(smem_ptr) % 16 == 0) && "Shared memory pointer not 16B aligned for cp.async");
//     assert((reinterpret_cast<uintptr_t>(gmem_ptr) % 16 == 0) && "Global memory pointer not 16B aligned for cp.async");

// #ifdef CP_ASYNC_SM80_ENABLED
//     uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
//     asm volatile(
//         "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" 
//         ::"r"(smem_int_ptr),
//           "l"(gmem_ptr),
//           "n"(16),
//           "r"(16)
//     );
// #else
//     #pragma unroll
//     for (int i = 0; i < 128 / sizeof(FP8); i++) {
//         smem_ptr[i] = gmem_ptr[i];
//     }
// #endif
// }

template <typename FP8, int N>
__device__ __forceinline__ void load_smem_raw3(FP8 dst[N], const FP8* src) {
#pragma unroll
  for (int i = 0; i < N; ++i) {
    dst[i] = src[i];
  }
}


template <typename FP8>
__device__ __forceinline__ void load_smem_raw2(FP8 *smem_ptr, FP8 const *gmem_ptr) {
    // 检查对齐
    bool smem_aligned = (reinterpret_cast<uintptr_t>(smem_ptr) % 16) == 0;
    bool gmem_aligned = (reinterpret_cast<uintptr_t>(gmem_ptr) % 16) == 0;
    // assert alignment
    assert(smem_aligned && "Shared memory pointer not 16B aligned for cp.async");
    assert(gmem_aligned && "Global memory pointer not 16B aligned for cp.async");
    // assert((!gmem_aligned) && "Global memory pointer not 16B aligned for cp.async");

#ifdef CP_ASYNC_SM80_ENABLED
    if (smem_aligned && gmem_aligned) {
        // 对齐正常，使用 cp.async 异步加载 128B
        // printf("Using cp.async for 128B load\n");
        uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
        asm volatile(
            "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" 
            ::"r"(smem_int_ptr),      // shared memory destination (int ptr)
              "l"(gmem_ptr),          // global memory source (64-bit ptr)
              "n"(16),                // 每个线程16B，8线程128B
              "r"(16)                 // zero-fill (no predication)
        );
    } else {
        // fallback: 普通加载（同步，逐元素拷贝）
        #pragma unroll
        for (int i = 0; i < 128 / sizeof(FP8); i++) {
            smem_ptr[i] = gmem_ptr[i];
        // #pragma unroll
        // for (int i = 0; i < 128 / sizeof(FP8); i++) {
        //     if (threadIdx.x == 0 && blockIdx.x == 0) {
        //         printf("DEBUG: gmem_ptr=%p smem_ptr=%p\n", gmem_ptr, smem_ptr);
        //     }
        //     // 额外检查是否越界
        //     if ((reinterpret_cast<const char*>(gmem_ptr + i) <
        //         reinterpret_cast<const char*>(gmem_base)) ||
        //         (reinterpret_cast<const char*>(gmem_ptr + i) >= 
        //         reinterpret_cast<const char*>(gmem_base) + gmem_size)) {
        //         asm("brkpt;");  // 触发断点
        //     }
        //     smem_ptr[i] = gmem_ptr[i];
        }
    }
#else
    // 架构 < SM80，直接用同步拷贝
    printf("Using direct copy for 128B load\n");
    #pragma unroll
    for (int i = 0; i < 128 / sizeof(FP8); i++) {
        smem_ptr[i] = gmem_ptr[i];
    }
#endif
}



// Define FP8 type (choose format as needed)
// using fp8_t = type::fp8_e4m3_t;  // or type::fp8_e5m2_t

template <typename T,   // e.g., fp16
        //   typename FP8, // e.g., uint8_t
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int O_STRIDE = OUTPUT_SIZE,
          typename FP8 = __uint8_t, // e.g., uint8_t
          int K_PIPE_MAX = 3>
__device__ __forceinline__ void linear_kernel_fp8_weight(  
    void const *input_ptr,             // fp16 input
    void const *weight_fp8_ptr,        // fp8 weight
    void const *weight_scale_ptr,      // fp16 scale (per group)
    void const *residual_ptr,          // fp16 residual
    void *output_ptr,                  // fp16 output
    // void const *weight_scale_ptr = nullptr,      // fp16 scale (per group)
    bool residual = true) {
      // Global memory pointer alignment checks
  assert_aligned(input_ptr, alignof(T), "input_ptr");
  assert_aligned(weight_fp8_ptr, alignof(FP8), "weight_fp8_ptr");
  assert_aligned(weight_scale_ptr, alignof(T), "weight_scale_ptr");
  if (residual_ptr) assert_aligned(residual_ptr, alignof(T), "residual_ptr");
  assert_aligned(output_ptr, alignof(T), "output_ptr");

  constexpr int CHUNK_SIZE = 16 / sizeof(T);
  constexpr int OUTPUT_ATOM_SIZE = OUTPUT_SIZE <= 128 ? OUTPUT_SIZE : 128;
  constexpr int NUM_OUTPUT_ATOMS = OUTPUT_SIZE / OUTPUT_ATOM_SIZE;
  constexpr int TILE_SIZE = 128;
  constexpr int FORLOOP_RANGE = REDUCTION_SIZE / TILE_SIZE;

  constexpr int NUM_CHUNKS_A = BATCH_SIZE * TILE_SIZE / CHUNK_SIZE;
  constexpr int NUM_CHUNKS_B = TILE_SIZE * OUTPUT_ATOM_SIZE / CHUNK_SIZE;
  constexpr int NUM_CHUNKS_C = BATCH_SIZE * OUTPUT_ATOM_SIZE / CHUNK_SIZE;

  constexpr int CHUNKS_PER_ROW_A = TILE_SIZE / CHUNK_SIZE;
  constexpr int CHUNKS_PER_COL_B = TILE_SIZE / CHUNK_SIZE;
  constexpr int CHUNKS_PER_ROW_C = OUTPUT_ATOM_SIZE / CHUNK_SIZE;

  constexpr int log2_CHUNK_SIZE = log2_constexpr(CHUNK_SIZE);
  constexpr int log2_CHUNKS_PER_ROW_A = log2_constexpr(CHUNKS_PER_ROW_A);
  constexpr int log2_CHUNKS_PER_COL_B = log2_constexpr(CHUNKS_PER_COL_B);
  constexpr int log2_CHUNKS_PER_ROW_C = log2_constexpr(CHUNKS_PER_ROW_C);

  // FP8 specific
  constexpr int CHUNK_SIZE_W = 16 / sizeof(FP8); // for FP8 load chunks
  constexpr int NUM_CHUNKS_B_FP8 = TILE_SIZE * OUTPUT_ATOM_SIZE / CHUNK_SIZE_W;
  constexpr int CHUNKS_PER_COL_B_FP8 = TILE_SIZE / CHUNK_SIZE_W;
  constexpr int log2_CHUNK_SIZE_W = log2_constexpr(CHUNK_SIZE_W);
  constexpr int log2_CHUNKS_PER_COL_B_FP8 = log2_constexpr(CHUNKS_PER_COL_B_FP8);


  // using SM80_16x8x16_F16F16F16F16_TNX2 = 16X16X16
  constexpr int NUM_WARPS_N =
      OUTPUT_ATOM_SIZE / 16 <= 4 ? OUTPUT_ATOM_SIZE / 16 : 4;
  constexpr int NUM_WARPS_K = 4 / NUM_WARPS_N;

  constexpr int NUM_ITERS_M = 1;
  constexpr int NUM_ITERS_N = OUTPUT_ATOM_SIZE / NUM_WARPS_N / 16;
  constexpr int NUM_ITERS_K = TILE_SIZE / NUM_WARPS_K / 16;

  constexpr int log2_NUM_WARPS_N = log2_constexpr(NUM_WARPS_N);
  constexpr int log2_NUM_ITERS_K = log2_constexpr(NUM_ITERS_K);

  int warp_idx = warp_id();
  int warp_row = warp_idx >> log2_NUM_WARPS_N;
  int warp_col = warp_idx & (NUM_WARPS_N - 1);
  int lane_idx = lane_id();

  T const *__restrict__ d_input = static_cast<T const *>(input_ptr);
  // Weight pointer is FP8
  FP8 const *__restrict__ d_weight_fp8 = static_cast<FP8 const *>(weight_fp8_ptr);
  T const *__restrict__ d_weight_scale = static_cast<T const *>(weight_scale_ptr);
  T const *__restrict__ d_residual =
      residual ? static_cast<T const *>(residual_ptr) : nullptr;
  T *__restrict__ d_output = static_cast<T *>(output_ptr);

  using InputDmem = dmem_row_const<T, BATCH_SIZE, TILE_SIZE, REDUCTION_SIZE>;
  using WeightDmemFP8 =
      dmem_col_const<FP8, TILE_SIZE, OUTPUT_ATOM_SIZE, REDUCTION_SIZE>;

  using ResidualDmem =
      dmem_row_const<T, BATCH_SIZE, OUTPUT_ATOM_SIZE, O_STRIDE>;
  using OutputDmem = dmem_row<T, BATCH_SIZE, OUTPUT_ATOM_SIZE, O_STRIDE>;

  // using ScaleDmem =
  //     dmem_col_const<T, 1, OUTPUT_ATOM_SIZE / GROUP_SIZE, REDUCTION_SIZE / GROUP_SIZE>;
  // ScaleDmem weight_scale_dmem(d_weight_scale);


  InputDmem input_dmem(d_input);
  WeightDmemFP8 weight_dmem_fp8(d_weight_fp8);  // <-- FP8 weight DMEM
  ResidualDmem residual_dmem(d_residual);
  OutputDmem output_dmem(d_output);

  extern __shared__ char smem[];




  // STensors' offsets
  constexpr size_t ZERO_BUFFER_OFFSET = 0;
  // sizeof(T) * 8

  constexpr size_t SHARED_INPUT_BUFFER_OFFSET =
      ZERO_BUFFER_OFFSET + sizeof(T) * 8;
  // sizeof(T) * K_PIPE_MAX * BATCH_SIZE * TILE_SIZE

  constexpr size_t SHARED_WEIGHT_BUFFER_OFFSET =
      SHARED_INPUT_BUFFER_OFFSET +
      sizeof(T) * K_PIPE_MAX * BATCH_SIZE * TILE_SIZE;
  // sizeof(T) * K_PIPE_MAX * TILE_SIZE * OUTPUT_ATOM_SIZE

  constexpr size_t SHARED_RESIDUAL_OFFSET =
      SHARED_WEIGHT_BUFFER_OFFSET +
      sizeof(T) * K_PIPE_MAX * TILE_SIZE * OUTPUT_ATOM_SIZE;
  // sizeof(T) * BATCH_SIZE * OUTPUT_ATOM_SIZE

  constexpr size_t MM_INTERMEDIATE_OFFSET =
      SHARED_RESIDUAL_OFFSET + sizeof(T) * BATCH_SIZE * OUTPUT_ATOM_SIZE;
  // sizeof(T) * NUM_WARPS_K * BATCH_SIZE * OUTPUT_ATOM_SIZE

  constexpr size_t SHARED_OUTPUT_OFFSET =
      MM_INTERMEDIATE_OFFSET +
      sizeof(T) * NUM_WARPS_K * BATCH_SIZE * OUTPUT_ATOM_SIZE;
  // sizeof(T) * BATCH_SIZE * OUTPUT_ATOM_SIZE

  assert_aligned(smem + SHARED_INPUT_BUFFER_OFFSET, alignof(T), "shared_input_buffer");
  assert_aligned(smem + SHARED_WEIGHT_BUFFER_OFFSET, alignof(T), "shared_weight_buffer");
  if (residual) assert_aligned(smem + SHARED_RESIDUAL_OFFSET, alignof(T), "shared_residual");
  assert_aligned(smem + MM_INTERMEDIATE_OFFSET, alignof(T), "mm_intermediate");
  assert_aligned(smem + SHARED_OUTPUT_OFFSET, alignof(T), "shared_output");

  // zero buffer
  T *zero_buf = (T *)(smem + ZERO_BUFFER_OFFSET);
  vec_zero_t<T, 8>::fill_zero(zero_buf);

  // copy
  T *shared_input_buffer = (T *)(smem + SHARED_INPUT_BUFFER_OFFSET);
  T *shared_weight_buffer = (T *)(smem + SHARED_WEIGHT_BUFFER_OFFSET);

  // residual
  T *shared_residual =
      residual ? (T *)(smem + SHARED_RESIDUAL_OFFSET) : nullptr;

  // intermediate
  T *mm_intermediate = (T *)(smem + MM_INTERMEDIATE_OFFSET);

  // output
  T *shared_output = (T *)(smem + SHARED_OUTPUT_OFFSET);

  // define the swizzle mode
  using ZeroBufferSmem = smem_row<T, 0, 0, 0, 1, 8, 8>;
  using InputSmem = smem_row<T, 0, 0, 0, BATCH_SIZE, TILE_SIZE, TILE_SIZE>;
  using InputBufferSmem =
      smem_row<T, 0, 0, 0, K_PIPE_MAX * BATCH_SIZE, TILE_SIZE, TILE_SIZE>;
  using WeightSmem =
      smem_col<T, 3, 3, 3, TILE_SIZE, OUTPUT_ATOM_SIZE, TILE_SIZE>;
  using WeightBufferSmem =
      smem_col<T, 3, 3, 3, TILE_SIZE, K_PIPE_MAX * OUTPUT_ATOM_SIZE, TILE_SIZE>;
  using OutputSmem =
      smem_row<T, 0, 0, 0, BATCH_SIZE, OUTPUT_ATOM_SIZE, OUTPUT_ATOM_SIZE>;
  using MatMulIntermediateSmem = smem_row<T,
                                          0,
                                          0,
                                          0,
                                          NUM_WARPS_K * BATCH_SIZE,
                                          OUTPUT_ATOM_SIZE,
                                          OUTPUT_ATOM_SIZE>;

  ZeroBufferSmem zero_buffer(zero_buf);

  InputSmem input_smem(shared_input_buffer);
  WeightSmem weight_smem(shared_weight_buffer);

  OutputSmem residual_smem(shared_residual);

  MatMulIntermediateSmem mm_intermediate_smem(mm_intermediate);

  OutputSmem output_smem(shared_output);

  for (int output_atom_idx = 0; output_atom_idx < NUM_OUTPUT_ATOMS;
       output_atom_idx++,
           d_weight_fp8 += OUTPUT_ATOM_SIZE * REDUCTION_SIZE,
           d_residual = residual ? d_residual + OUTPUT_ATOM_SIZE : nullptr,
           d_output += OUTPUT_ATOM_SIZE) {
    weight_dmem_fp8.set_ptr(d_weight_fp8);
    residual_dmem.set_ptr(d_residual);
    output_dmem.set_ptr(d_output);

    InputBufferSmem input_buffer_smem(shared_input_buffer);
    WeightBufferSmem weight_buffer_smem(shared_weight_buffer);

    if (residual) {
#pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_C; i += NUM_THREADS) {
        int row = i >> log2_CHUNKS_PER_ROW_C;
        int col = (i & (CHUNKS_PER_ROW_C - 1)) << log2_CHUNK_SIZE;
        load_smem(residual_smem(row, col), residual_dmem(row, col));
      }
    }

#pragma unroll
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; k_pipe++) {
      // note that it's < K_PIPE_MAX - 1, because the last k_pipe is handled separately!!!
#pragma unroll
      // __syncthreads();
      // if(k_pipe == K_PIPE_MAX - 1) {
      //             printf("k_pipe #1 = %d, output_atom_idx = %d\n", k_pipe, output_atom_idx);
      // }
      for (int i = threadIdx.x; i < NUM_CHUNKS_A; i += NUM_THREADS) {
        int src_row = i >> log2_CHUNKS_PER_ROW_A;
        // i / 2^x where x s.t. 2^x is the closest to CHUNKS_PER_ROW_A = (TILE_SIZE / CHUNK_SIZE) 
        int dst_row = src_row + ((k_pipe + 1) * BATCH_SIZE);
        int dst_col = (i & (CHUNKS_PER_ROW_A - 1)) << log2_CHUNK_SIZE;
        int src_col = dst_col + (k_pipe << log2_constexpr(TILE_SIZE));
        load_smem(input_buffer_smem(dst_row, dst_col),
                  input_dmem(src_row, src_col));
      }
      // if (threadIdx.x == 0 && blockIdx.x == 0) {
      //   printf("Debug: tid = %d, k_pipe = %d\n", threadIdx.x, k_pipe);
      // }
    //   printf("Reached before __syncthreads(): tid = %d, bid = %d\n", threadIdx.x, blockIdx.x);
      // __syncthreads();
      // if (threadIdx.x == 0 && blockIdx.x == 0) {
      //   printf("Debug 783: tid = %d, k_pipe = %d\n", threadIdx.x, k_pipe);
      // }
      // __syncthreads();
    //   printf("Reached after __syncthreads(): tid = %d, bid = %d\n", threadIdx.x, blockIdx.x);

#pragma unroll
      // for (int i = threadIdx.x; i < NUM_CHUNKS_B; i += NUM_THREADS) {
      //   int dst_row = (i & (CHUNKS_PER_COL_B - 1)) << log2_CHUNK_SIZE;
      //   int src_row = dst_row + (k_pipe << log2_constexpr(TILE_SIZE));
      //   int src_col = i >> log2_CHUNKS_PER_COL_B;
      //   int dst_col =
      //       src_col + ((k_pipe + 1) << log2_constexpr(OUTPUT_ATOM_SIZE));
      for (int i = threadIdx.x; i < NUM_CHUNKS_B_FP8; i += NUM_THREADS) {
        int dst_row = (i & (CHUNKS_PER_COL_B_FP8 - 1)) << log2_CHUNK_SIZE_W;
        int src_row = dst_row + (k_pipe << log2_constexpr(TILE_SIZE));
        // dst_row = 2 * dst_row; // Adjust for FP8 chunk size
        int src_col = i >> log2_CHUNKS_PER_COL_B_FP8;
        int dst_col =
            src_col + ((k_pipe + 1) << log2_constexpr(OUTPUT_ATOM_SIZE));

        // if(k_pipe == K_PIPE_MAX - 1) {
        //   printf("k_pipe #2 = %d, output_atom_idx = %d\n", k_pipe, output_atom_idx);
        // }


        // Load FP8 weights from GMEM into registers
        FP8 fp8_val[CHUNK_SIZE_W];

        // void* gmem_addr = static_cast<void*>(weight_dmem_fp8(src_row, src_col));
        // uintptr_t addr_val = reinterpret_cast<uintptr_t>(gmem_addr);
        // if (addr_val % alignof(FP8) != 0) {
        //     printf("Misaligned GMEM FP8 address: %p (mod %ld)\n", gmem_addr, addr_val % alignof(FP8));
        //     assert(false && "GMEM FP8 load address not aligned");
        // }

        // print weight_dmem_fp8(src_row, src_col)
        // if (threadIdx.x == 0 && blockIdx.x == 0) 
        //     printf("Loading FP8 weights from GMEM: src_row = %d, src_col = %d, dst_row = %d, dst_col = %d, val = %f\n",
        //         src_row, src_col, dst_row, dst_col, fp8_to_float(weight_dmem_fp8.at((size_t)src_row, (size_t)src_col)));
        // __syncthreads();

        // load_smem_raw2(fp8_val, weight_dmem_fp8(src_row, src_col)); // raw FP8 load
        load_smem_raw3<FP8, CHUNK_SIZE_W>(fp8_val, weight_dmem_fp8(src_row, src_col)); // raw FP8 load
        // todo: this function needs to be defined

        // printf("Debug 829: tid = %d, k_pipe = %d, dst_row = %d, dst_col = %d\n", threadIdx.x, k_pipe, dst_row, dst_col);

        // if (threadIdx.x == 0 && blockIdx.x == 0) {
        //     printf("Debug 1: tid = %d, k_pipe = %d, dst_row = %d, dst_col = %d\n", threadIdx.x, k_pipe, dst_row, dst_col);
        // }
        // __syncthreads();

        // if(k_pipe == K_PIPE_MAX - 1) {
        //     printf("k_pipe = %d, dst_row = %d, dst_col = %d\n", k_pipe, dst_row, dst_col);
        // }

        // Fetch scale (group-wise): assume GROUP_SIZE along K dimension
        // int group_idx = src_row / GROUP_SIZE;
        // T scale = weight_scale_dmem(group_idx, src_col);
        T scale =static_cast<T>(1.0f); // For simplicity, assume scale is 1.0f
        // Convert FP8 → FP16 in shared memory

        #pragma unroll
        for (int k = 0; k < CHUNK_SIZE_W; k++) {
            assert((reinterpret_cast<uintptr_t>(weight_buffer_smem(dst_row + k, dst_col)) % alignof(T) == 0) &&
            "weight_buffer_smem not aligned");
            // assert((reinterpret_cast<uintptr_t>(weight_buffer_smem(dst_row + k, dst_col)) % 16 == 0) &&
            // "weight_buffer_smem not aligned");

            T fp16_val = static_cast<T>(fp8_to_float(fp8_val[k])) * scale; // unquantize
            // todo: fp8_to_float needs to be defined
            if ((reinterpret_cast<uintptr_t>(weight_buffer_smem(dst_row + k, dst_col)) % alignof(T)) != 0) {
                    printf("Misaligned shared mem write! ptr = %p\n", weight_buffer_smem(dst_row + k, dst_col));
                }
            // if ((reinterpret_cast<uintptr_t>(weight_buffer_smem(dst_row + k, dst_col)) % 16) != 0) {
            //         printf("Misaligned shared mem write! ptr = %p\n", weight_buffer_smem(dst_row + k, dst_col));
            //     }
            weight_buffer_smem(dst_row + k, dst_col)[0] = fp16_val;

        }

        // // 先把 16 个 FP8 转成 16 个 FP16，放到寄存器数组 tmp16[]
        // T tmp16[CHUNK_SIZE_W]; // 16
        // #pragma unroll
        // for (int k = 0; k < CHUNK_SIZE_W; ++k) {
        //   tmp16[k] = static_cast<T>(fp8_to_float(fp8_val[k])) * scale;
        // }

        // // 以 8 为步长写回（每次对齐 16B）
        // #pragma unroll
        // for (int kk = 0; kk < CHUNK_SIZE_W; kk += CHUNK_SIZE /*8*/) {
        //   T* dst = weight_buffer_smem(dst_row + kk, dst_col); // 这里 row = 0 或 8，满足块对齐
        //   // 保证 dst 16B 对齐（建议保留断言/打印 mod16）：
        //   // assert(((uintptr_t)dst % 16) == 0);
        //   #pragma unroll
        //   for (int t = 0; t < CHUNK_SIZE; ++t) {
        //     dst[t] = tmp16[kk + t];
        //   }
        // }


        // __syncthreads();
        // if (threadIdx.x == 0 && blockIdx.x == 0) {
        //     printf("Debug 2: tid = %d, k_pipe = %d, dst_row = %d, dst_col = %d\n", threadIdx.x, k_pipe, dst_row, dst_col);
        // }
      }
      cp_async_fence();
    }

    // __syncthreads();

    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //   printf("Debug839: tid = %d, k_pipe = %d\n", threadIdx.x, K_PIPE_MAX - 1);
    // }

    // accumulator
    alignas(16) float s_frag[NUM_ITERS_M][NUM_ITERS_N][8];
    for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
#pragma unroll
      for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
        clear_8_floats(s_frag[m][n]);
      }
    }

    for (int for_idx = 0; for_idx < FORLOOP_RANGE; for_idx++) {
      // copy
      if (for_idx + K_PIPE_MAX - 1 < FORLOOP_RANGE) {
#pragma unroll
        for (int i = threadIdx.x; i < NUM_CHUNKS_A; i += NUM_THREADS) {
          int row = i >> log2_CHUNKS_PER_ROW_A;
          int dst_col = (i & (CHUNKS_PER_ROW_A - 1)) << log2_CHUNK_SIZE;
          int src_col = dst_col + ((for_idx + K_PIPE_MAX - 1)
                                   << log2_constexpr(TILE_SIZE));
          load_smem(input_buffer_smem(row, dst_col), input_dmem(row, src_col));
        }
#pragma unroll
        for (int i = threadIdx.x; i < NUM_CHUNKS_B_FP8; i += NUM_THREADS) {
          int dst_row = (i & (CHUNKS_PER_COL_B_FP8 - 1)) << log2_CHUNK_SIZE_W;
          // we assume that CHUNKS_PER_COL_B_FP8 - 1 is the power of 2, so we can use bitwise operations, which is equivalent to taking the lower bits
          // of the index
          int src_row = dst_row + ((for_idx + K_PIPE_MAX - 1)
                                   << log2_constexpr(TILE_SIZE));
            // dst_row = 2 * dst_row; // Adjust for FP8 chunk size
          int col = i >> log2_CHUNKS_PER_COL_B_FP8;
          // load_smem(weight_buffer_smem(dst_row, col),
          //           weight_dmem_fp8(src_row, col));
          // Load FP8 weights from GMEM into registers

          alignas(16) FP8 fp8_val[CHUNK_SIZE_W];
          load_smem_raw3<FP8, CHUNK_SIZE_W>(fp8_val, weight_dmem_fp8(src_row, col)); // raw FP8 load

          // Fetch scale (group-wise): assume GROUP_SIZE along K dimension
          // int group_idx = src_row / GROUP_SIZE;
          // T scale = weight_scale_dmem(group_idx, src_col);
          T scale = static_cast<T>(1.0f); // For simplicity, assume scale is 1.0f

          // Convert FP8 → FP16 in shared memory
          #pragma unroll
          for (int k = 0; k < CHUNK_SIZE_W; k++) {
              T fp16_val = static_cast<T>(fp8_to_float(fp8_val[k])) * scale; // unquantize

            //   T* smem_ptr = weight_buffer_smem(dst_row + k, dst_col);
            //     uintptr_t smem_addr = reinterpret_cast<uintptr_t>(smem_ptr);
            //     if (smem_addr % alignof(T) != 0) {
            //         printf("Misaligned SHMEM store: dst_row=%d, dst_col=%d, addr=%p (mod %ld)\n",
            //             dst_row + k, dst_col, smem_ptr, smem_addr % alignof(T));
            //         assert(false && "Shared memory write not aligned");
            //     }
            //     smem_ptr[0] = fp16_val;


              weight_buffer_smem(dst_row + k, col)[0] = fp16_val;
          }

        }
        cp_async_fence();
        cp_async_wait<K_PIPE_MAX - 1>();
      } else if (for_idx + K_PIPE_MAX - 1 == FORLOOP_RANGE) {
        cp_async_wait<0>();
      }

      // rotate the buffers
      input_buffer_smem.set_ptr(shared_input_buffer +
                                BATCH_SIZE * TILE_SIZE *
                                    ((for_idx + 1) % K_PIPE_MAX));
      input_smem.set_ptr(shared_input_buffer +
                         BATCH_SIZE * TILE_SIZE * ((for_idx + 1) % K_PIPE_MAX));
      weight_buffer_smem.set_ptr(shared_weight_buffer +
                                 TILE_SIZE * OUTPUT_ATOM_SIZE *
                                     ((for_idx + 1) % K_PIPE_MAX));
      weight_smem.set_ptr(shared_weight_buffer +
                          TILE_SIZE * OUTPUT_ATOM_SIZE *
                              ((for_idx + 1) % K_PIPE_MAX));
      __syncthreads();
      // To rotate shared memory buffers in a pipeline fashion, allowing the reuse of a limited number of buffers (K_PIPE_MAX) instead of allocating a large amount of shared memory.
      // Idea: After processing the current tile, switch the pointers to the next preloaded tile in shared memory.

      uint32_t a_frag[4], b_frag[4];
      for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
        int m_row = (lane_idx & 0xF);
        bool is_valid = (m_row < BATCH_SIZE);
#pragma unroll
        for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
          int n_col = (n << (4 + log2_NUM_WARPS_N)) + (warp_col << 4) +
                      ((lane_idx >> 4) << 3) + (lane_idx & 0x7);
#pragma unroll
          for (uint32_t k = 0; k < NUM_ITERS_K; k++) {
            int m_col = (warp_row << (4 + log2_NUM_ITERS_K)) + (k << 4) +
                        ((lane_idx >> 4) << 3);
            int n_row = (warp_row << (4 + log2_NUM_ITERS_K)) + (k << 4) +
                        (((lane_idx & 0xF) >> 3) << 3);
            T *src_ptr =
                is_valid ? input_smem(m_row, m_col) : zero_buffer(0, 0);
            ldsm(src_ptr, a_frag);
            ldsm(weight_smem(n_row, n_col), b_frag);
            mma_m16n16k16_bf16bf16bf32(
                s_frag[m][n], a_frag, b_frag, s_frag[m][n]);
          }
        }
      }
      __syncthreads();
    }

    // write back to shared memory
    for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
#pragma unroll
      for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
#pragma unroll
        for (uint32_t i = 0; i < 4; i++) {
          int row_in_warp = (lane_idx >> 2) + ((i & 0x1) << 3);
          if (row_in_warp < BATCH_SIZE) {
            int col = (n << (4 + log2_NUM_WARPS_N)) + (warp_col << 4) +
                      ((lane_idx & 0x3) << 1) + ((i >> 1) << 3);
            mm_intermediate_smem.at(warp_row + row_in_warp, col) =
                bfloat16(s_frag[m][n][(i << 1)]);
            mm_intermediate_smem.at(warp_row + row_in_warp, col + 1) =
                bfloat16(s_frag[m][n][(i << 1) | 0x1]);
          }
        }
      }
    }
    __syncthreads();

    if (NUM_WARPS_K > 1) {
      reduction_sum_row<decltype(output_smem), decltype(mm_intermediate_smem)>(
          output_smem, mm_intermediate_smem);
      __syncthreads();
    }

#pragma unroll
    for (int row = 0; row < BATCH_SIZE; row++) {
#pragma unroll
      for (int i = threadIdx.x; i < OUTPUT_ATOM_SIZE; i += NUM_THREADS) {
        T val = NUM_WARPS_K > 1 ? output_smem.at(row, i)
                                : mm_intermediate_smem.at(row, i);
        output_dmem.at(row, i) =
            residual ? val + residual_smem.at(row, i) : val;
      }
    }
    if (output_atom_idx + 1 < NUM_OUTPUT_ATOMS) {
      __syncthreads();
    }
  }
}

}
