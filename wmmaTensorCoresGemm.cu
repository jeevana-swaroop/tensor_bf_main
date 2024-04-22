/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// CUDA sample demonstrating a GEMM computation using the Warp Matrix Multiply
// and Accumulate API introduced in CUDA 9.

// In this program, the compute_gemm kernel computes the result of a matrix
// multiplication and addition: D = alpha * A * B + beta * C. The dimensions of
// both C and D matrices are M_GLOBAL x N_GLOBAL. The A matrix is M_GLOBAL x
// K_GLOBAL (row-major), the B matrix is K_GLOBAL x N_GLOBAL (column-major). In
// that kernel, each CTA computes one 128 x 128 tile of the resulting matrix per
// iteration. When the tile is computed, the CTA stores it to the global memory
// and begins a new iteration, selecting a new 128 x 128 tile to compute.
// Each CTA consists of eight warps. For the 128 x 128 tile, each warp computes
// eight 16 x 16 subtiles, organized in a 2 x 4 two-dimensional array. Warps
// compute the 16 x 16 subtiles using nvcuda::wmma::mma_sync operations by
// moving through the K_GLOBAL dimension of the A and B matrices and
// accumulating the intermediate result in the local thread state.

// There are a number of simple optimizations used in the algorithm:
// - The CTA copies the 128 x 128 tile of the C matrix from the global memory to
//   shared memory. After that is done, each warp loads the C matrix fragments
//   from shared memory, thus avoiding a random global memory access.
// - On each internal iteration, the CTA copies a portion of the A and B
//   matrices from global memory to shared memory. After that, all warps in the
//   CTA reuse the A and B data from shared memory, thus reducing the number of
//   data copies from global memory.
// - The portions of the A and B matrices are stored in shared memory with an
//   additional padding (skew) to reduce the number of shared memory access bank
//   conflicts.
//   (See a detailed explanation near the SKEW_HALF macro definition.)
// - When the CTA finishes computing the tiles of the resulting matrix, each
//   warp stores its subtiles to shared memory. The CTA then copies the shared
//   memory contents to global memory, again avoiding redundant random global
//   memory  accesses.
// - Note that the CTA tile size is chosen to maximize the GPU register
//   utilization, but carefully enough to avoid local memory use.

#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#include "params.h"
#include "wmmaTensorCoresGemm.cuh"

#if COMPUTE_PREC != 64

using namespace nvcuda;

__host__ void init_host_matrices(half *a, half *b, float *c) {
    for (unsigned  int i = 0; i < M_GLOBAL; i++) {
        for (unsigned int j = 0; j < K_GLOBAL; j++) {
            a[i * K_GLOBAL + j] = (half)(rand() % 3);
        }
    }

    for (unsigned int i = 0; i < N_GLOBAL; i++) {
        for (unsigned int j = 0; j < K_GLOBAL; j++) {
            b[i * K_GLOBAL + j] = (half)(rand() % 3);
        }
    }

    for (unsigned int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
        c[t] = static_cast<float>(rand() % 3);
    }
}

__global__ void compute_gemm(const half *A, const half *B, const float *C,
                             float *D, float alpha, float beta) {
    extern __shared__ half shmem[][CHUNK_K * K + SKEW_HALF];

    // Warp and lane identification.
    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const unsigned int laneId = threadIdx.x % WARP_SIZE;

    // Offset in shared memory from which the B matrix is stored.
    const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

    // This pointer is used to access the C and D matrix tiles this warp computes.
    float *shmem_warp_tile_ptr = (float *)&shmem[0][0] +
                                 (warpId / 2) * SHMEM_STRIDE * K * 2 +
                                 (warpId % 2) * SHMEM_OFFSET;

    // This pointer is used to stream the C and D matrices block-wide tile to and
    // from shared memory.
    float *shmem_warp_stream_ptr =
            (float *)&shmem[0][0] + warpId * SHMEM_STRIDE * K;

    // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
    // each tile computation. Technically this is not generally correct (may
    // result in a loss of precision). Zero still needs to be specially handled
    // though.
    beta /= alpha;

    // Each CTA slides along the 128 x 128 tiles from the top left corner of the
    // matrix to the right and down, and selects the next tile to compute. Once
    // there's no such tile, all warps in this CTA exit.
    for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
        const unsigned int block_tile_i =
                ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
        const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

        // Stop when there are no more D matrix tiles to compute in this CTA.
        if (block_tile_i >= M_TILES) {
            break;
        }

        // This warp's pointer to the C matrix data to copy memory from to shared
        // memory.
        const size_t gmem_idx =
                (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
        const float *src_gmem_warp_stream_ptr = &C[gmem_idx];

        // Stream multiple C tiles to shared memory.
#pragma unroll
        for (int i = 0; i < K; i++) {
            typedef int4 copy_t;

            *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
                    *((copy_t *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) +
                      laneId);
        }

        __syncthreads();

        // These fragments will accumulate the result of A and B matrix fragment
        // multiplications along the K_GLOBAL dimension.
        wmma::fragment<wmma::accumulator, M, N, K, float> c[WARP_COL_TILES]
        [WARP_ROW_TILES];

        // Load the C matrix tiles into fragments from shared memory.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
                const float *tile_ptr =
                        shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

                wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
            }
        }

        __syncthreads();

        // Scale the C matrix.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
                for (int t = 0; t < c[i][j].num_elements; t++) {
                    c[i][j].x[t] *= beta;
                }
            }
        }

        // Select what warp copies what matrix to shared memory.
        // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
        const half *warp_ptr = (warpId < 4) ? (&A[block_tile_i * M * K_GLOBAL] +
                                               M * K_GLOBAL * (warpId % 4) * 2)
                                            : (&B[block_tile_j * N * K_GLOBAL] +
                                               N * K_GLOBAL * (warpId % 4) * 2);

        // Go through the global K dimension by a fixed step at a time.
#pragma unroll
        for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
            // Copy slices of the A and B matrices to shared memory.
            // The first half of the warps in the CTA copy the A matrix, the rest copy
            // the B matrix.
            size_t shmem_idx =
                    warpId < (WARPS_PER_BLOCK / 2)
                    ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
                    : (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

            // First half of the warp copies the first row / column of the matrix,
            // the second half of the warp copies the next.
            int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * K +
                                      (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) +
                             (laneId % CHUNK_COPY_LINE_LANES);

            // Shift the second half of the warp to the next row / column in the
            // shared memory.
            shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
            for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2;
                 i++) {
                // Copy 16 bytes at once in each lane.
                *((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
                        *lane_ptr;

                // Advance the global memory pointer and the shared memory index.
                lane_ptr =
                        (int4 *)((half *)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
                shmem_idx += CHUNK_COPY_LINES_PER_WARP;
            }

            __syncthreads();

            // Compute a grid of C matrix tiles in each warp.
#pragma unroll
            for (int k_step = 0; k_step < CHUNK_K; k_step++) {
                wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major>
                        a[WARP_COL_TILES];
                wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major>
                        b[WARP_ROW_TILES];

#pragma unroll
                for (int i = 0; i < WARP_COL_TILES; i++) {
                    size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
                    const half *tile_ptr = &shmem[shmem_idx_a][k_step * K];

                    wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_HALF);

#pragma unroll
                    for (int j = 0; j < WARP_ROW_TILES; j++) {
                        if (i == 0) {
                            // Load the B matrix fragment once, because it is going to be
                            // reused against the other A matrix fragments.
                            size_t shmem_idx_b = shmem_idx_b_off +
                                                 (WARP_ROW_TILES * N) * (warpId % 2) +
                                                 (j * N);
                            const half *tile_ptr = &shmem[shmem_idx_b][k_step * K];

                            wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_HALF);
                        }

                        wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                    }
                }
            }

            __syncthreads();
        }

        // Store the D fragments to shared memory.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
                // Uniform, point-wise transformations of ALL fragment elements by ALL
                // threads in the warp are well-defined even though element indices
                // within fragment storage are not defined.
                for (int t = 0; t < c[i][j].num_elements; t++) c[i][j].x[t] *= alpha;

                float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

                wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
            }
        }

        __syncthreads();

        // Now that shared memory contains all the D tiles, stream them to global
        // memory.
        float *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
        for (int i = 0; i < K; i++) {
            *((int4 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
                    *((int4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
        }

        __syncthreads();
    }
}

// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16.
//  3) Neither A nor B are transposed.
// Note: This is a less performant version of the compute_gemm kernel. It is
// designed for
//       demonstration purposes only to show the CUDA WMMA API use without
//       relying on availability of the shared memory.
__global__ void simple_wmma_gemm(half *a, half *b, float *c, float *d, int m_ld,
                                 int n_ld, int k_ld, float alpha, float beta) {
    // Leading dimensions. Packed with no transpositions.
    int lda = k_ld;
    int ldb = k_ld;
    int ldc = n_ld;

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
            a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>
            b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over k
    for (int i = 0; i < k_ld; i += WMMA_K) {
        int aCol = i;
        int aRow = warpM * WMMA_M;
        int bCol = warpN * N;
        int bRow = i;

        // Bounds checking
        if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
            wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Load in the current value of c, scale it by beta, and add this our result
    // scaled by alpha
    int cCol = warpN * WMMA_N;
    int cRow = warpM * WMMA_M;

    if (cRow < m_ld && cCol < n_ld) {
        wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc,
                               wmma::mem_row_major);

        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // Store the output
        wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc,
                                wmma::mem_row_major);
    }
}

__host__ void matMultiplyOnHost(half *A, half *B, float *C, float alpha,
                                float beta, int numARows, int numAColumns,
                                int numBRows, int numBColumns, int numCRows,
                                int numCColumns) {
    for (int i = 0; i < numCRows; i++) {
        for (int j = 0; j < numCColumns; j++) {
            float temp = 0.0;

            for (int k = 0; k < numAColumns; k++) {
                temp += (float)A[i * numAColumns + k] * (float)B[j * numBRows + k];
            }

            C[i * numCColumns + j] = temp * alpha + beta * C[i * numCColumns + j];
        }
    }
}

#endif // COMPUTE_PREC != 64