#ifndef DMMATENSORCORESGEMM_CUH
#define DMMATENSORCORESGEMM_CUH

#include "params.h"

#if COMPUTE_PREC == 64

__global__ void compute_dgemm(const double *A, const double *B, const double *C, double *D, double alpha, double beta);

__global__ void compute_dgemm_async_copy(const double *A, const double *B, const double *C, double *D, double alpha, double beta);

__global__ void compute_dgemm_cg_async_copy(const double *A, const double *B, const double *C, double *D, double alpha, double beta);

__global__ void simple_wmma_gemm(double *a, double *b, double *c, double *d, int m_ld, int n_ld, int k_ld, double alpha, double beta);

__host__ void matMultiplyOnHost(double *A, double *B, double *C,
                                float alpha, float beta,
                                int numARows, int numAColumns,
                                int numBRows, int numBColumns,
                                int numCRows, int numCColumns);

#endif

#endif //DMMATENSORCORESGEMM_CUH
