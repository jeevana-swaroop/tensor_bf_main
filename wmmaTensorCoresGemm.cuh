#ifndef TENSOR_BF_WMMATENSORCORESGEMM_H
#define TENSOR_BF_WMMATENSORCORESGEMM_H

#include "params.h"

#if COMPUTE_PREC != 64

__global__ void compute_gemm(const half* A, const half* B, const float* C,
                             float* D, float alpha, float beta);

__global__ void simple_wmma_gemm(half* a, half* b, float* c, float* d, int m_ld,
                                 int n_ld, int k_ld, float alpha, float beta);

__host__ void matMultiplyOnHost(half* A, half* B, float* C, float alpha,
                                float beta, int numARows, int numAColumns,
                                int numBRows, int numBColumns, int numCRows,
                                int numCColumns);

#endif

#endif //TENSOR_BF_WMMATENSORCORESGEMM_H
