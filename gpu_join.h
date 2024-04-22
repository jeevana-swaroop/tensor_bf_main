#ifndef GPU_JOIN_H
#define GPU_JOIN_H

#include <cuda_fp16.h>

#include "params.h"

void fillMatrixDiagonalHalf(half** iMatrix, half value);

void fillMatrixDiagonalDouble(double** iMatrix, double value);

//void GPUJoinMainBruteForceCutlass(
//        unsigned int searchMode,
//        unsigned int device,
//        double* dataset,
//        unsigned int* nbQueryPoints,
//        double* epsilon,
//        uint64_t* totalNeighbors);

//void CPUJoinMainBruteForce(
//        unsigned int searchMode,
//        double* dataset,
//        unsigned int* nbQueryPoints,
//        double* epsilon,
//        uint64_t* totalNeighbors);

void GPUJoinMainBruteForceNvidia(
        unsigned int searchMode,
        unsigned int device,
        INPUT_DATA_TYPE* dataset,
        INPUT_DATA_TYPE* datasetTranspose,
        unsigned int* nbQueryPoints,
        ACCUM_TYPE* epsilon,
        uint64_t* totalNeighbors);

void GPUJoinMainBruteForce(
    unsigned int searchMode,
    unsigned int device,
    INPUT_DATA_TYPE* dataset,
    unsigned int* nbQueryPoints,
    ACCUM_TYPE* epsilon,
    uint64_t* totalNeighbors);

#endif