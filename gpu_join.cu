#include <stdio.h>
#include <math.h>
#include <iostream>
#include <utility>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

//#include "cutlass/cutlass.h"
//#include "cutlass/gemm/device/gemm.h"

//#include "cutlass/util/command_line.h"
//#include "cutlass/util/host_tensor.h"
//#include "cutlass/util/reference/device/gemm.h"
//#include "cutlass/util/reference/host/tensor_compare.h"
//#include "cutlass/util/reference/host/tensor_copy.h"
//#include "cutlass/util/reference/host/tensor_fill.h"
//#include "cutlass/util/tensor_view_io.h"

//#include "dmmaTensorCoresGemm.cuh"
//#include "wmmaTensorCoresGemm.cuh"

#include "kernel_join.h"
#include "gpu_join.h"
#include "utils.h"
#include "params.h"


void fillMatrixDiagonalHalf(half** iMatrix, half value)
{
    for (unsigned int i = 0; i < 16; ++i)
    {
        for (unsigned int j = 0; j < 16; ++j)
        {
            if (i == j)
            {
                (*iMatrix)[i * 16 + j] = value;
            } else {
                (*iMatrix)[i * 16 + j] = (half)0.0;
            }
        }
    }
}


// "Special" identity matrix for double precision tensor cores:
// | 1 0 0 0 |
// | 0 1 0 0 |
// | 0 0 1 0 |
// | 0 0 0 1 |
// | 1 0 0 0 |
// | 0 1 0 0 |
// | 0 0 1 0 |
// | 0 0 0 1 |
void fillMatrixDiagonalDouble(double** iMatrix, double value)
{
    // Fill with zeros
    for (unsigned int i = 0; i < 8; ++i)
    {
        for (unsigned int j = 0; j < 4; ++j)
        {
            (*iMatrix)[i * 4 + j] = 0.0;
        }
    }

    // Complete with the specified value
    (*iMatrix)[0] = value;
    (*iMatrix)[5] = value;
    (*iMatrix)[10] = value;
    (*iMatrix)[15] = value;

    (*iMatrix)[16] = value;
    (*iMatrix)[21] = value;
    (*iMatrix)[26] = value;
    (*iMatrix)[31] = value;
}


//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\


void GPUJoinMainBruteForceNvidia(
    unsigned int searchMode,
    unsigned int device,
    INPUT_DATA_TYPE* dataset,
    INPUT_DATA_TYPE* datasetTranspose,
    unsigned int* nbQueryPoints,
    ACCUM_TYPE* epsilon,
    uint64_t* totalNeighbors)
{
//    cudaSetDevice(device);
//    unsigned int nbBlockTmp = ceil((1.0 * (*nbQueryPoints)) / (1.0 * BLOCKSIZE));
//
//    INPUT_DATA_TYPE* dev_datasetA;
//    cudaErrCheck(cudaMalloc((void **)&dev_datasetA, sizeof(INPUT_DATA_TYPE) * (COMPUTE_DIM) * (*nbQueryPoints)));
//    cudaErrCheck(cudaMemcpy(dev_datasetA, dataset, sizeof(INPUT_DATA_TYPE) * (COMPUTE_DIM) * (*nbQueryPoints), cudaMemcpyHostToDevice));
//
//    INPUT_DATA_TYPE* dev_datasetB;
//    cudaErrCheck(cudaMalloc((void **)&dev_datasetB, sizeof(INPUT_DATA_TYPE) * (COMPUTE_DIM) * (*nbQueryPoints)));
//    cudaErrCheck(cudaMemcpy(dev_datasetB, datasetTranspose, sizeof(INPUT_DATA_TYPE) * (COMPUTE_DIM) * (*nbQueryPoints), cudaMemcpyHostToDevice));
////    transposeDataset<<<nbBlockTmp, BLOCKSIZE>>>(dev_datasetA, dev_datasetB, (*nbQueryPoints));
//
//    #if INPUT_DATA_PREC != COMPUTE_PREC
//        COMPUTE_TYPE* dev_datasetAltA;
//        cudaErrCheck(cudaMalloc((void **)&dev_datasetAltA, sizeof(COMPUTE_TYPE) * (COMPUTE_DIM) * (*nbQueryPoints)));
//        convertDataset<<<nbBlockTmp, BLOCKSIZE>>>(dev_datasetA, dev_datasetAltA, (*nbQueryPoints));
//
//        COMPUTE_TYPE* dev_datasetAltB;
//        cudaErrCheck(cudaMalloc((void **)&dev_datasetAltB, sizeof(COMPUTE_TYPE) * (COMPUTE_DIM) * (*nbQueryPoints)));
//        convertDataset<<<nbBlockTmp, BLOCKSIZE>>>(dev_datasetB, dev_datasetAltB, (*nbQueryPoints));
//    #endif
//
//    ACCUM_TYPE* dev_preComputedCoordinates;
//    cudaErrCheck(cudaMalloc((void **)&dev_preComputedCoordinates, sizeof(ACCUM_TYPE) * (*nbQueryPoints)));
//    #if INPUT_DATA_PREC != COMPUTE_PREC
//        preComputedSquaredCoordinatesComplete<<<nbBlockTmp, BLOCKSIZE>>>(dev_datasetAltA, dev_preComputedCoordinates, (*nbQueryPoints));
//    #else
//        preComputedSquaredCoordinatesComplete<<<nbBlockTmp, BLOCKSIZE>>>(dev_datasetA, dev_preComputedCoordinates, (*nbQueryPoints));
//    #endif
//
//    ACCUM_TYPE* dev_matrixResult;
////    cudaErrCheck(cudaMalloc((void **)&dev_matrixResult, sizeof(double) * (*nbQueryPoints) * (*nbQueryPoints)));
//    cudaErrCheck(cudaMallocManaged((void**)&dev_matrixResult, (uint64_t)(sizeof(ACCUM_TYPE)) * (uint64_t)(*nbQueryPoints) * (uint64_t)(*nbQueryPoints)));
//    fillResultMatrix<<<nbBlockTmp, BLOCKSIZE>>>(dev_preComputedCoordinates, dev_matrixResult, (*nbQueryPoints));
//
////    double* dev_matrixD;
////    cudaErrCheck(cudaMalloc((void **)&dev_matrixD, sizeof(double) * (*nbQueryPoints) * (*nbQueryPoints)));
////    cudaErrCheck(cudaMemset(dev_matrixD, 0, sizeof(double) * (*nbQueryPoints) * (*nbQueryPoints)));
//
//    unsigned long long* cnt = new unsigned long long;
//    unsigned long long* dev_cnt;
//    cudaErrCheck(cudaMalloc((void**)&dev_cnt, sizeof(unsigned long long)));
//
//    ACCUM_TYPE* dev_epsilon;
//    cudaErrCheck(cudaMalloc((void**)&dev_epsilon, sizeof(ACCUM_TYPE)));
//    cudaErrCheck(cudaMemcpy(dev_epsilon, epsilon, sizeof(ACCUM_TYPE), cudaMemcpyHostToDevice));
//
//    #if COMPUTE_PREC == 64
//    size_t SHMEM_SZ = max(sizeof(COMPUTE_TYPE) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_DOUBLE) * 2,
//                          M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(ACCUM_TYPE));
//    #else
//    size_t SHMEM_SZ = max(sizeof(COMPUTE_TYPE) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_HALF) * 2,
//                    M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(ACCUM_TYPE));
//    #endif
//
//    const ACCUM_TYPE alpha = (-2.0f);
//    const ACCUM_TYPE beta = 1.0f;
//
//    cudaDeviceProp deviceProp;
//    cudaErrCheck(cudaGetDeviceProperties(&deviceProp, device));
//
//    dim3 gridDim;
//    dim3 blockDim;
//
////     blockDim.x must be a multple of warpSize
////     128x4 means we have 16 warps and a block computes a 64x64 output tile
//    blockDim.x = 128;
//    blockDim.y = 4;
//
//    gridDim.x = (M_GLOBAL + (WMMA_M * blockDim.x / 32 - 1)) /
//                (WMMA_M * blockDim.x / 32);
//    gridDim.y = (N_GLOBAL + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
//
//    printf("Computing... using simple_wmma_gemm kernel\n");
//    simple_wmma_gemm<<<gridDim, blockDim>>>(dev_datasetAltA, dev_datasetAltA, dev_matrixResult, dev_matrixResult,
//                                            M_GLOBAL, N_GLOBAL, K_GLOBAL, alpha, beta);
//
//    cudaDeviceSynchronize();
//
//    finishResultMatrix<<<nbBlockTmp, BLOCKSIZE>>>(dev_preComputedCoordinates, dev_matrixResult, (*nbQueryPoints), dev_cnt, dev_epsilon);
//
//    cudaDeviceSynchronize();
//
//    cudaErrCheck(cudaMemcpy(cnt, dev_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
//    (*totalNeighbors) = (*cnt);
//
////    if (deviceProp.sharedMemPerMultiprocessor >= SHMEM_SZ)
////    {
////        #if COMPUTE_PREC == 64
////            cudaErrCheck(cudaFuncSetAttribute(compute_dgemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
////            compute_dgemm<<<deviceProp.multiProcessorCount*2, THREADS_PER_BLOCK, SHMEM_SZ>>>(dev_datasetA,dev_datasetB, dev_matrixResult,
////                                                                                             dev_matrixResult, alpha, beta);
////        #else
////            cudaErrCheck(cudaFuncSetAttribute(compute_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
////            #if INPUT_DATA_PREC != COMPUTE_PREC
////                compute_gemm<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK, SHMEM_SZ>>>(dev_datasetAltA, dev_datasetAltA, dev_matrixResult,
////                                                                                                 dev_matrixResult, alpha, beta);
////            #else
////                compute_gemm<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK, SHMEM_SZ>>>(dev_datasetA, dev_datasetB, dev_matrixResult,
////                                                                                                 dev_matrixResult, alpha, beta);
////            #endif
////        #endif
////
////        cudaDeviceSynchronize();
////
////        finishResultMatrix<<<nbBlockTmp, BLOCKSIZE>>>(dev_preComputedCoordinates, dev_matrixResult, (*nbQueryPoints), dev_cnt, dev_epsilon);
////
////        cudaDeviceSynchronize();
////
////        cudaErrCheck(cudaMemcpy(cnt, dev_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
////        (*totalNeighbors) = (*cnt);
////    } else {
////        fprintf(stderr, "[GPU] ~ Error: Insufficient shared memory per multiprocessor. Requested: %zu, available: %zu\n", SHMEM_SZ, deviceProp.sharedMemPerMultiprocessor);
////        fprintf(stderr, "[GPU] ~ Error: Cannot compute this kernel on this GPU.\n");
////    }
}


void GPUJoinMainBruteForce(
    unsigned int searchMode,
    unsigned int device,
    INPUT_DATA_TYPE* dataset,
    unsigned int* nbQueryPoints,
    ACCUM_TYPE* epsilon,
    uint64_t* totalNeighbors)
{
    cudaSetDevice(device);

    INPUT_DATA_TYPE* dev_dataset;
    cudaErrCheck(cudaMalloc((void **)&dev_dataset, sizeof(INPUT_DATA_TYPE) * (COMPUTE_DIM) * ((*nbQueryPoints) + ADDITIONAL_POINTS)));
    cudaErrCheck(cudaMemcpy(dev_dataset, dataset, sizeof(INPUT_DATA_TYPE) * (COMPUTE_DIM) * ((*nbQueryPoints) + ADDITIONAL_POINTS), cudaMemcpyHostToDevice));

    unsigned int nbBlockTmp = ceil((1.0 * ((*nbQueryPoints) + ADDITIONAL_POINTS)) / (1.0 * BLOCKSIZE));
    #if INPUT_DATA_PREC != COMPUTE_PREC
        COMPUTE_TYPE* dev_datasetAlt;
        cudaErrCheck(cudaMalloc((void **)&dev_datasetAlt, sizeof(COMPUTE_TYPE) * (COMPUTE_DIM) * ((*nbQueryPoints) + ADDITIONAL_POINTS)));
        convertDataset<<<nbBlockTmp, BLOCKSIZE>>>(dev_dataset, dev_datasetAlt, ((*nbQueryPoints) + ADDITIONAL_POINTS));
    #endif

    COMPUTE_TYPE* identityMatrix;
    COMPUTE_TYPE* dev_identityMatrix;
    #if COMPUTE_PREC == 16
        identityMatrix = new COMPUTE_TYPE[16 * 16];
        fillMatrixDiagonalHalf(&identityMatrix, 1.0);
        cudaErrCheck(cudaMalloc((void**)&dev_identityMatrix, sizeof(half) * 16 * 16));
        cudaErrCheck(cudaMemcpy(dev_identityMatrix, identityMatrix, sizeof(half) * 16 * 16, cudaMemcpyHostToDevice));
    #else
        identityMatrix = new COMPUTE_TYPE[8 * 4];
        fillMatrixDiagonalDouble(&identityMatrix, 1.0);
        cudaErrCheck(cudaMalloc((void**)&dev_identityMatrix, sizeof(double) * 8 * 4));
        cudaErrCheck(cudaMemcpy(dev_identityMatrix, identityMatrix, sizeof(double) * 8 * 4, cudaMemcpyHostToDevice));
    #endif

    ACCUM_TYPE* dev_preComputedSquaredCoordinates;
    if (SM_TENSOR_MQ == searchMode)
    {
        #if COMPUTE_PREC != INPUT_DATA_PREC
            cudaErrCheck(cudaMalloc((void**)&dev_preComputedSquaredCoordinates, sizeof(ACCUM_TYPE) * ((*nbQueryPoints) + ADDITIONAL_POINTS) * (COMPUTE_DIM / 16)));
            preComputedSquaredCoordinates<<<nbBlockTmp, BLOCKSIZE>>>(dev_datasetAlt, dev_preComputedSquaredCoordinates, ((*nbQueryPoints) + ADDITIONAL_POINTS));
        #else
            cudaErrCheck(cudaMalloc((void**)&dev_preComputedSquaredCoordinates, sizeof(ACCUM_TYPE) * ((*nbQueryPoints) + ADDITIONAL_POINTS) * (COMPUTE_DIM / 4)));
            preComputedSquaredCoordinates<<<nbBlockTmp, BLOCKSIZE>>>(dev_dataset, dev_preComputedSquaredCoordinates, ((*nbQueryPoints) + ADDITIONAL_POINTS));
        #endif
    }

    unsigned int* dev_nbQueryPoints;
    cudaErrCheck(cudaMalloc((void**)&dev_nbQueryPoints, sizeof(unsigned int)));
    cudaErrCheck(cudaMemcpy(dev_nbQueryPoints, nbQueryPoints, sizeof(unsigned int), cudaMemcpyHostToDevice));

    unsigned long long* cnt = new unsigned long long;
    unsigned long long* dev_cnt;
    cudaErrCheck(cudaMalloc((void**)&dev_cnt, sizeof(unsigned long long)));

    ACCUM_TYPE* dev_epsilon;
    cudaErrCheck(cudaMalloc((void**)&dev_epsilon, sizeof(ACCUM_TYPE)));
    cudaErrCheck(cudaMemcpy(dev_epsilon, epsilon, sizeof(ACCUM_TYPE), cudaMemcpyHostToDevice));

    const unsigned int tensorBlockSize = WARP_PER_BLOCK * WARP_SIZE;

    switch (searchMode)
    {
        case SM_GPU:
        {
            const unsigned int nbBlock = ceil((1.0 * (*nbQueryPoints)) / (1.0 * BLOCKSIZE));
            #if INPUT_DATA_PREC != COMPUTE_PREC
                distanceCalculationBruteForceCuda<<<nbBlock, BLOCKSIZE>>>(dev_nbQueryPoints, dev_datasetAlt, dev_epsilon, dev_cnt);
            #else
                distanceCalculationBruteForceCuda<<<nbBlock, BLOCKSIZE>>>(dev_nbQueryPoints, dev_dataset, dev_epsilon, dev_cnt);
            #endif
            break;
        }
        case SM_CUDA_ALT:
        {
            const unsigned int nbBlock = ceil((1.0 * (*nbQueryPoints)) / (1.0 * BLOCKSIZE));
            #if INPUT_DATA_PREC != COMPUTE_PREC
                distanceCalculationBruteForceCudaAlt<<<nbBlock, BLOCKSIZE>>>(dev_nbQueryPoints, dev_datasetAlt, dev_epsilon, dev_cnt);
            #else
                distanceCalculationBruteForceCudaAlt<<<nbBlock, BLOCKSIZE>>>(dev_nbQueryPoints, dev_dataset, dev_epsilon, dev_cnt);
            #endif
            break;
        };
        case SM_TENSOR:
        {
            const unsigned int nbBlock = ceil(((1.0 * (*nbQueryPoints)) / (1.0 * tensorBlockSize)) * WARP_SIZE);
            distanceCalculationBruteForceTensorBasic<<<nbBlock, tensorBlockSize>>>(dev_nbQueryPoints, dev_datasetAlt, dev_epsilon, dev_identityMatrix, dev_cnt);
            break;
        }
        case SM_TENSOR_MQ:
        {
            #if COMPUTE_PREC == 16
//                const unsigned int nbBlock = ceil(((1.0 * (*nbQueryPoints)) / (1.0 * tensorBlockSize)) * WARP_SIZE);
                const unsigned int nbBlock = ceil(((WARP_SIZE/Q * (*nbQueryPoints)) / (1.0 * tensorBlockSize)));
                distanceCalculationBruteForceTensorHalf<<<nbBlock, tensorBlockSize>>>(dev_nbQueryPoints, dev_datasetAlt, dev_epsilon, dev_cnt, dev_preComputedSquaredCoordinates);
            #else
                const unsigned int nbBlock = ceil((1.0 * (*nbQueryPoints) * 4) / (1.0 * tensorBlockSize));
                distanceCalculationBruteForceTensorDouble<<<nbBlock, tensorBlockSize>>>(dev_nbQueryPoints, dev_dataset, dev_epsilon, dev_cnt, dev_preComputedSquaredCoordinates);
            #endif
            break;
        }
        default:
        {
            std::cerr << "[GPU] ~ Error: Unknown search mode\n";
            exit(1);
        }
    }

    cudaDeviceSynchronize();

    cudaErrCheck(cudaMemcpy(cnt, dev_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    (*totalNeighbors) = (*cnt);
}