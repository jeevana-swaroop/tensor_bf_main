#ifndef PARAMS_H
#define PARAMS_H

#define INPUT_DATA_DIM 90
#define COMPUTE_DIM 96

#define BLOCKSIZE 256
#define WARP_PER_BLOCK 4

#define INPUT_DATA_PREC 64
#define COMPUTE_PREC 16
#define ACCUM_PREC 32

#define Q 16
#define C 16

//#define CUDA_ALT 0
//#define CUDA_KAHAN 0

#define ADDITIONAL_POINTS 15

/*********************************************************************/
/*                  Nvidia DMMA Tensor Cores GEMM                    */
/*********************************************************************/

#ifndef CPU_DEBUG
// Set this to 1 to verify the correctness of the GPU-computed matrix.
#define CPU_DEBUG 0
#endif

#ifndef SHARED_MEMORY_LIMIT_64K
// Set this to 0 to use more than 64 Kb of shared memory to cache data, to
// improve the performance of the computations on GPU.
// Note that you need a GPU that can have more than 64 Kb of shared memory
// per multiprocessor.
#define SHARED_MEMORY_LIMIT_64K 0
#endif

// MMA matrix tile dimensions.
#if COMPUTE_PREC == 64
    #define M 8
    #define N 8
    #define K 4
    #define WMMA_M 8
    #define WMMA_N 8
    #define WMMA_K 4
#else
    #define M 16
    #define N 16
    #define K 16
    #define WMMA_M 16
    #define WMMA_N 16
    #define WMMA_K 16
#endif

// GEMM configuration.
#define M_TILES 128
#define N_TILES 128
#define K_TILES 1

#define M_GLOBAL (M * M_TILES)
#define N_GLOBAL (N * N_TILES)
#define K_GLOBAL (K * K_TILES)

#define C_LAYOUT wmma::mem_row_major

// Implementation constants.
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#if SHARED_MEMORY_LIMIT_64K
// With only 64 Kb shared memory available, we can fit 8x16-tile chunks of each
// the A and B matrix data, that are (M = 8) * (K = 4) * 8 * (CHUNK_K = 16) * sizeof(double) = 32 Kb each
// But we cannot account the 4 Kb total skew overhead, without which the performance
// would be severely impacted. So we choose to reduce the chunk size in half,
// i.e. the amount of A and B matrix data we cache in shared memory.
// Accordingly, this doubles the number of outer iterations across the global K
// dimension, which only slightly impacts the performance.
    #if COMPUTE_PREC == 64
        #define CHUNK_K 8
    #else
        #define CHUNK_K 4
    #endif
#else
    #if COMPUTE_PREC == 64
        #define CHUNK_K 16
    #else
        #define CHUNK_K 8
    #endif
#endif

#if COMPUTE_PREC == 64
    #define CHUNK_LINE_BYTES (CHUNK_K * K * sizeof(double))
#else
    #define CHUNK_LINE_BYTES (CHUNK_K * K * sizeof(half))
#endif
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

//#define BLOCK_ROW_WARPS 4
//#define BLOCK_COL_WARPS 2
#if COMPUTE_PREC == 64
    #define BLOCK_ROW_WARPS 4
    #define BLOCK_COL_WARPS 2
#else
    #define BLOCK_ROW_WARPS 2
    #define BLOCK_COL_WARPS 4
#endif

//#define WARP_ROW_TILES 2
//#define WARP_COL_TILES 4
#if COMPUTE_PREC == 64
    #define WARP_ROW_TILES 2
    #define WARP_COL_TILES 4
#else
    #define WARP_ROW_TILES 4
    #define WARP_COL_TILES 2
#endif

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define GLOBAL_MEM_STRIDE N_GLOBAL

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)

// The macro below is used to shift rows of the A matrix and columns of the B matrix
// in shared memory to minimize possible bank conflicts.
// Before performing the nvcuda::wmma::mma_sync operation, the warp must load the matrix
// data using the nvcuda::wmma::load_matrix_sync operation. Although the memory access pattern
// is not specified for that function, each lane in the warp can read one or multiple matrix
// elements from different matrix rows or columns.
// For shared memory, such access can result in bank conflicts if different rows / columns
// of the matrix map to the same bank. By shifting each row and column by a few bytes, we
// make sure that they map to different banks, thus reducing the number of possible bank
// conflicts.
// The number of 4 eight-byte "double" elements is chosen as the minimum possible shift because
// we must keep each row and column 256-bit aligned, as required by nvcuda::wmma::load_matrix_sync.
#define SKEW_DOUBLE 2
#define SKEW_HALF 16

#define checkKernelErrors(expr) do {                                                        \
    expr;                                                                                   \
                                                                                            \
    cudaError_t __err = cudaGetLastError();                                                 \
    if (__err != cudaSuccess) {                                                             \
        printf("Line %d: '%s' failed: %s\n", __LINE__, # expr, cudaGetErrorString(__err));  \
        abort();                                                                            \
    }                                                                                       \
} while(0)


/*********************************************************************/
/*                 Code below should not be modified                 */
/*********************************************************************/


#define NB_ARGS_MAX 5
#define FILENAME_ARG 1
#define EPSILON_ARG 2
#define SEARCHMODE_ARG 3
#define DEVICE_ARG 4

#define SM_GPU 1                // CUDA cores normal
#define SM_CUDA_ALT 2           // CUDA cores alt
#define SM_TENSOR 3             // Tensor cores 1Q
#define SM_TENSOR_MQ 4          // Tensor cores 16Q
#define SM_NVIDIA 5             // Nvidia Tensor cores sample solution
#define SM_CPU 6                // CPU algorithm

#define WARP_SIZE 32

#if INPUT_DATA_PREC == 16
    #define INPUT_DATA_TYPE half
#else
    #if INPUT_DATA_PREC == 32
        #define INPUT_DATA_TYPE float
    #else
        #define INPUT_DATA_TYPE double
    #endif
#endif

#if COMPUTE_PREC == 16
    #define COMPUTE_TYPE half
#else
    #if COMPUTE_PREC == 32
        #define COMPUTE_TYPE float
    #else
        #define COMPUTE_TYPE double
    #endif
#endif

#if ACCUM_PREC == 16
    #define ACCUM_TYPE half
#else
    #if ACCUM_PREC == 32
        #define ACCUM_TYPE float
    #else
        #define ACCUM_TYPE double
    #endif
#endif

#endif
