#include <math.h>
#include <stdio.h>

#include "kernel_join.h"
#include "params.h"

#include <mma.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>

using namespace nvcuda;
using namespace cooperative_groups;


__global__ void printMatrix(double *matrix, unsigned int nbElements) {
    for (unsigned int i = 0; i < nbElements; ++i) {
        for (unsigned int j = 0; j < COMPUTE_DIM; ++j) {
            printf("%f ", matrix[i * COMPUTE_DIM + j]);
        }
        printf("\n");
    }
}

__global__ void printMatrixTranspose(double *matrix, unsigned int size, unsigned int nbElements) {
    for (unsigned int i = 0; i < COMPUTE_DIM; ++i) {
        for (unsigned int j = 0; j < nbElements; ++j) {
            printf("%f ", matrix[i * size + j]);
        }
        printf("\n");
    }
}


__global__ void printMatrixResult(double *matrix, unsigned int size, unsigned int nbElements) {
    for (unsigned int i = 0; i < nbElements; ++i) {
        for (unsigned int j = 0; j < COMPUTE_DIM; ++j) {
            printf("%f ", matrix[i * size + j]);
        }
        printf("\n");
    }
}


__global__ void convertDataset(
        INPUT_DATA_TYPE *in,
        COMPUTE_TYPE *out,
        unsigned int nbPoints) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nbPoints) {
        for (unsigned int i = 0; i < COMPUTE_DIM; ++i) {
            out[tid * COMPUTE_DIM + i] = (COMPUTE_TYPE)(in[tid * COMPUTE_DIM + i]);
        }
    }
}


__global__ void preComputedSquaredCoordinates(
        COMPUTE_TYPE *dataset,
        ACCUM_TYPE *preComputeCoordinates,
        unsigned int nbQueryPoints) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nbQueryPoints <= tid) {
        return;
    }

#if ACCUM_PREC == 64
    double accum[4];
    for (unsigned int i = 0; i < COMPUTE_DIM; i += 4)
    {
        accum[0] = dataset[tid * COMPUTE_DIM + i] * dataset[tid * COMPUTE_DIM + i];
        accum[1] = dataset[tid * COMPUTE_DIM + i + 1] * dataset[tid * COMPUTE_DIM + i + 1];
        accum[2] = dataset[tid * COMPUTE_DIM + i + 2] * dataset[tid * COMPUTE_DIM + i + 2];
        accum[3] = dataset[tid * COMPUTE_DIM + i + 3] * dataset[tid * COMPUTE_DIM + i + 3];
        preComputeCoordinates[tid * (COMPUTE_DIM / 4) + (i / 4)] = accum[0] + accum[1] + accum[2] + accum[3];
    }
#else
    //		float accum[16];
    for (unsigned int i = 0; i < COMPUTE_DIM; i += 16) {
        float accum = 0.0;
#pragma unroll
        for (unsigned int j = 0; j < 16; ++j) {
            accum +=
                    __half2float(dataset[tid * COMPUTE_DIM + i + j]) * __half2float(dataset[tid * COMPUTE_DIM + i + j]);
        }
        preComputeCoordinates[tid * (COMPUTE_DIM / 16) + (i / 16)] = accum;
    }
#endif
}


__global__ void preComputedSquaredCoordinatesComplete(
        COMPUTE_TYPE *dataset,
        ACCUM_TYPE *preComputeCoordinates,
        unsigned int nbQueryPoints) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nbQueryPoints <= tid) {
        return;
    }

    ACCUM_TYPE accum = 0.0;
    for (unsigned int i = 0; i < COMPUTE_DIM; ++i) {
        accum += (ACCUM_TYPE) (dataset[tid * COMPUTE_DIM + i]) * (ACCUM_TYPE) (dataset[tid * COMPUTE_DIM + i]);
    }
    preComputeCoordinates[tid] = accum;
}


__global__ void transposeDataset(
        COMPUTE_TYPE *inputDataset,
        COMPUTE_TYPE *outputDataset,
        unsigned int nbQueryPoints) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (nbQueryPoints <= tid) {
        return;
    }

    for (unsigned int i = 0; i < COMPUTE_DIM; ++i) {
        outputDataset[tid * COMPUTE_DIM + i] = inputDataset[i * COMPUTE_DIM + tid];
    }
}


__global__ void fillResultMatrix(
        ACCUM_TYPE *preComputedSquaredCoordinates,
        ACCUM_TYPE *resultMatrix,
        unsigned int nbQueryPoints) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nbQueryPoints <= tid) {
        return;
    }

    for (unsigned int i = 0; i < nbQueryPoints; ++i) {
        resultMatrix[i * nbQueryPoints + tid] = preComputedSquaredCoordinates[tid];
    }
}


__global__ void finishResultMatrix(
        ACCUM_TYPE *preComputedSquaredCoordinates,
        ACCUM_TYPE *resultMatrix,
        unsigned int nbQueryPoints,
        unsigned long long *cnt,
        ACCUM_TYPE *epsilon) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nbQueryPoints <= tid) {
        return;
    }

    for (unsigned int i = 0; i < nbQueryPoints; ++i) {
        ACCUM_TYPE finalDistance = fabs(resultMatrix[i * nbQueryPoints + tid] + preComputedSquaredCoordinates[i]);

#if ACCUM_PREC == 16
        if (hsqrt(finalDistance) <= (*epsilon))
#else
        if (sqrt(finalDistance) <= (*epsilon))
#endif
        {
            unsigned int idx = atomicAdd(cnt, int(1));
        }
    }
}


//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\


__global__ void distanceCalculationBruteForceCuda(
        unsigned int *nbQueryPoints,
        COMPUTE_TYPE *dataset,
        ACCUM_TYPE *epsilon,
        unsigned long long *cnt) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if ((*nbQueryPoints) <= tid) {
        return;
    }

    COMPUTE_TYPE point[INPUT_DATA_DIM];
    for (unsigned int i = 0; i < INPUT_DATA_DIM; ++i) {
        point[i] = dataset[tid * COMPUTE_DIM + i];
    }

    for (unsigned int i = 0; i < (*nbQueryPoints); ++i) {
        ACCUM_TYPE accumDistance = 0.0;
        for (unsigned int j = 0; j < INPUT_DATA_DIM; ++j) {
            accumDistance += (ACCUM_TYPE) ((point[j] - dataset[i * COMPUTE_DIM + j]) *
                                           (point[j] - dataset[i * COMPUTE_DIM + j]));
        }

#if ACCUM_PREC == 16
        if(hsqrt(accumDistance) <= (*epsilon))
#else
        if (sqrt(accumDistance) <= (*epsilon))
#endif
        {
            unsigned int idx = atomicAdd(cnt, int(1));
        }
    }
}


__global__ void distanceCalculationBruteForceCudaAlt(
        unsigned int *nbQueryPoints,
        COMPUTE_TYPE *dataset,
        ACCUM_TYPE *epsilon,
        unsigned long long *cnt) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if ((*nbQueryPoints) <= tid) {
        return;
    }

    COMPUTE_TYPE point[INPUT_DATA_DIM];
    ACCUM_TYPE q2[INPUT_DATA_DIM];
    for (unsigned int i = 0; i < INPUT_DATA_DIM; ++i) {
        COMPUTE_TYPE queryCoord = dataset[tid * COMPUTE_DIM + i];
        point[i] = (COMPUTE_TYPE)(-2.0) * queryCoord;
        q2[i] = (ACCUM_TYPE) (queryCoord) * (ACCUM_TYPE) (queryCoord);
    }

    for (unsigned int i = 0; i < (*nbQueryPoints); ++i) {
        ACCUM_TYPE accumDistance = 0.0;
        for (unsigned int j = 0; j < INPUT_DATA_DIM; ++j) {
            COMPUTE_TYPE candidateCoord = dataset[i * COMPUTE_DIM + j];
            ACCUM_TYPE c2 = (ACCUM_TYPE) (candidateCoord * candidateCoord);
            accumDistance += (ACCUM_TYPE) (point[j] * candidateCoord) + q2[j] + c2;
        }

#if ACCUM_PREC == 16
        if(hsqrt(habs(accumDistance)) <= (*epsilon))
#else
        if (sqrt(fabs(accumDistance)) <= (*epsilon))
#endif
        {
            unsigned int idx = atomicAdd(cnt, int(1));
        }
    }
}


//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\


__global__ void distanceCalculationBruteForceTensorBasic(
        unsigned int *nbQueryPoints,
        COMPUTE_TYPE *dataset,
        ACCUM_TYPE *epsilon,
        COMPUTE_TYPE *identityMatrix,
        unsigned long long *cnt) {
    __shared__ half sharedArrayQueryPoints[WARP_PER_BLOCK * COMPUTE_DIM];
    __shared__ half sharedArrayResultFirstStep[WARP_PER_BLOCK * 16 * 16];
    __shared__ ACCUM_TYPE sharedArrayResultSecondStep[WARP_PER_BLOCK * 16 * 16];

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warpIdInGrid = tid / WARP_SIZE;
    unsigned int queryPoint = warpIdInGrid;
    unsigned int warpIdInBlock = threadIdx.x / WARP_SIZE;

    unsigned int sharedArrayResultOffset = warpIdInBlock * 16 * 16;

    if ((*nbQueryPoints) <= queryPoint) {
        return;
    }

    thread_block_tile<WARP_SIZE> warp = tiled_partition<WARP_SIZE>(this_thread_block());
    unsigned int halfWarpId = warp.thread_rank() / 16;
    unsigned int halfWarpThreadId = warp.thread_rank() % 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> matrixAFragment;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> matrixBFragment;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> identityFragment;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> firstStepAccumulator;
    wmma::fragment<wmma::accumulator, 16, 16, 16, ACCUM_TYPE> secondStepAccumulator;

    wmma::load_matrix_sync(identityFragment, identityMatrix, 16);

    for (unsigned int j = 0; j < COMPUTE_DIM; j += WARP_SIZE) {
        if ((j + warp.thread_rank()) < COMPUTE_DIM) {
            sharedArrayQueryPoints[warpIdInBlock * COMPUTE_DIM + j + warp.thread_rank()] = dataset[
                    queryPoint * COMPUTE_DIM + j + warp.thread_rank()];
        }
    }

    for (unsigned int i = 0; i < (*nbQueryPoints); i += 16) {
        unsigned int nbCandidatesLeft = (*nbQueryPoints) - i;
        unsigned int nbCandidatesCurrent = min(16, nbCandidatesLeft);

        wmma::fill_fragment(secondStepAccumulator, 0.0);

        for (unsigned int n = 0; n < COMPUTE_DIM; n += 16) {
            wmma::load_matrix_sync(matrixAFragment, sharedArrayQueryPoints + warpIdInBlock * COMPUTE_DIM + n, 0);
            wmma::load_matrix_sync(firstStepAccumulator, dataset + i * COMPUTE_DIM + n, COMPUTE_DIM,
                                   wmma::mem_row_major);
            for (int j = 0; j < firstStepAccumulator.num_elements; ++j) {
                firstStepAccumulator.x[j] = (half)(-1.0) * firstStepAccumulator.x[j];
            }

            wmma::mma_sync(firstStepAccumulator, matrixAFragment, identityFragment, firstStepAccumulator);
            wmma::store_matrix_sync(sharedArrayResultFirstStep + sharedArrayResultOffset, firstStepAccumulator, 16,
                                    wmma::mem_row_major);

            wmma::load_matrix_sync(matrixAFragment, sharedArrayResultFirstStep + sharedArrayResultOffset, 16);
            wmma::load_matrix_sync(matrixBFragment, sharedArrayResultFirstStep + sharedArrayResultOffset, 16);

            wmma::mma_sync(secondStepAccumulator, matrixAFragment, matrixBFragment, secondStepAccumulator);
        }

        wmma::store_matrix_sync(sharedArrayResultSecondStep + sharedArrayResultOffset, secondStepAccumulator, 16,
                                wmma::mem_row_major);
        if (warp.thread_rank() < 16 && warp.thread_rank() < nbCandidatesLeft) {
            ACCUM_TYPE resultDistance = sharedArrayResultSecondStep[sharedArrayResultOffset + warp.thread_rank() * 16 +
                                                                    warp.thread_rank()];

#if ACCUM_PREC == 16
            if(hsqrt(__habsresultDistance) <= (*epsilon))
#else
            if (sqrt(resultDistance) <= (*epsilon))
#endif
            {
                unsigned int tmpIdx = atomicAdd(cnt, int(1));
            }
        }
    }
}


__global__ void distanceCalculationBruteForceTensorHalf(
        unsigned int *nbQueryPoints,
        COMPUTE_TYPE *dataset,
        ACCUM_TYPE *epsilon,
        unsigned long long *cnt,
        ACCUM_TYPE *preComputedSquaredCoordinates) {
    __shared__ half sharedArrayQueryPoints[WARP_PER_BLOCK * Q * COMPUTE_DIM];
    __shared__ ACCUM_TYPE sharedArraySquaredQueries[WARP_PER_BLOCK * Q * (COMPUTE_DIM / 16)];
    __shared__ ACCUM_TYPE sharedArraySquaredCandidates[WARP_PER_BLOCK * C];
    __shared__ ACCUM_TYPE sharedArrayResultTmp[WARP_PER_BLOCK * Q * C];
    __shared__ ACCUM_TYPE sharedArrayResult[WARP_PER_BLOCK * Q * C];

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warpIdInGrid = tid / WARP_SIZE;
    unsigned int queryPoint = warpIdInGrid * Q;
    unsigned int warpIdInBlock = threadIdx.x / WARP_SIZE;

    if ((*nbQueryPoints) <= queryPoint) {
        return;
    }

    unsigned int sharedArrayQueryOffset = warpIdInBlock * Q * COMPUTE_DIM;
    unsigned int sharedArrayOffset = warpIdInBlock * Q * C;
    unsigned int sharedArraySquaredOffsetQuery = warpIdInBlock * Q * (COMPUTE_DIM / 16);

    thread_block_tile<32> warp = tiled_partition<32>(this_thread_block());
    thread_block_tile<16> tile16 = tiled_partition<16>(warp);
    thread_block_tile<8> tile8 = tiled_partition<8>(warp);

    unsigned int nbStepsToPage = ceil((1.0 * COMPUTE_DIM) / (1.0 * WARP_SIZE));
    unsigned int nbQueriesBatch = (queryPoint + Q > (*nbQueryPoints)) ? (*nbQueryPoints) - queryPoint : Q;

    for (unsigned int i = 0; i < nbQueriesBatch; ++i) {
        for (unsigned int j = 0; j < COMPUTE_DIM; j += 32) {
            if ((j + warp.thread_rank()) < COMPUTE_DIM) {
                sharedArrayQueryPoints[sharedArrayQueryOffset + i * COMPUTE_DIM + j + warp.thread_rank()] =
                        (COMPUTE_TYPE)(-2.0) * dataset[(queryPoint + i) * COMPUTE_DIM + j + warp.thread_rank()];

            }
        }
    }

    for (unsigned int i = nbQueriesBatch; i < Q; ++i) {
        for (unsigned int j = 0; j < COMPUTE_DIM; j += 32) {
            if ((j + warp.thread_rank()) < COMPUTE_DIM) {
                sharedArrayQueryPoints[sharedArrayQueryOffset + i * COMPUTE_DIM + j + warp.thread_rank()] = (half) 0.0;
            }
        }
    }


    if (warp.thread_rank() < Q) {
        for (unsigned int i = 0; i < (COMPUTE_DIM / 16); ++i) {
            if (warp.thread_rank() < nbQueriesBatch) {
                sharedArraySquaredQueries[sharedArraySquaredOffsetQuery + i * Q + warp.thread_rank()] =
                        preComputedSquaredCoordinates[(queryPoint + warp.thread_rank()) * (COMPUTE_DIM / 16) + i];
            } else {
                sharedArraySquaredQueries[sharedArraySquaredOffsetQuery + i * Q +
                                          warp.thread_rank()] = (ACCUM_TYPE) 0.0;
            }
        }
    }

//

    for (unsigned int i = 0; i < (*nbQueryPoints); i += C) {
        unsigned int nbCandidatesLeft = (*nbQueryPoints) - i;
        unsigned int nbCandidatesCurrent = min(C, nbCandidatesLeft);

        for (unsigned int j = 0; j < Q; j += WARP_SIZE / C) {
#if Q == 16
            {
                sharedArrayResult[sharedArrayOffset + (j + tile16.meta_group_rank()) * 16 + tile16.thread_rank()] = (ACCUM_TYPE) 0.0;
            }
#elif Q == 32
            {
                sharedArrayResult[sharedArrayOffset + (j + tile8.meta_group_rank()) * 8 +
                                  tile8.thread_rank()] = (ACCUM_TYPE) 0.0;
            }
#else
            {
                            sharedArrayResult[sharedArrayOffset + (j) * 32 + warp.thread_rank()] = (ACCUM_TYPE) 0.0;
                        }
#endif

        }

        for (unsigned int n = 0; n < COMPUTE_DIM; n += 16) {
            if (warp.thread_rank() < C) {
                if ((i + warp.thread_rank()) < (*nbQueryPoints)) {
                    unsigned int candidateId = i + warp.thread_rank();
                    sharedArraySquaredCandidates[warpIdInBlock * C + warp.thread_rank()] =
                            preComputedSquaredCoordinates[candidateId * (COMPUTE_DIM / 16) + (n / 16)];
                } else {
                    sharedArraySquaredCandidates[warpIdInBlock * C + warp.thread_rank()] = (ACCUM_TYPE) 0.0;
                }
            }

            wmma::fragment<wmma::matrix_a, Q, C, 16, half, wmma::row_major> matrixQ;
            wmma::fragment<wmma::matrix_b, Q, C, 16, half, wmma::col_major> matrixC;
            wmma::fragment<wmma::accumulator, Q, C, 16, ACCUM_TYPE> matrixC2;
            wmma::fragment<wmma::accumulator, Q, C, 16, ACCUM_TYPE> matrixQCC2;

            wmma::load_matrix_sync(matrixQ, sharedArrayQueryPoints + sharedArrayQueryOffset + n, COMPUTE_DIM);
            wmma::load_matrix_sync(matrixC, dataset + i * COMPUTE_DIM + n, COMPUTE_DIM);
            wmma::load_matrix_sync(matrixC2, sharedArraySquaredCandidates + (warpIdInBlock * C), 0,
                                   wmma::mem_row_major);
            wmma::fill_fragment(matrixQCC2, 0.0);

            wmma::mma_sync(matrixQCC2, matrixQ, matrixC, matrixC2);
            wmma::store_matrix_sync(sharedArrayResultTmp + sharedArrayOffset, matrixQCC2, C, wmma::mem_row_major);

            for (unsigned int j = 0; j < Q; j += WARP_SIZE / C) {
                unsigned int localId;

#if Q == 16
                {
                localId = sharedArrayOffset + (j + tile16.meta_group_rank()) * 16 + tile16.thread_rank();
                sharedArrayResult[localId] = sharedArrayResult[localId]
                                            + sharedArrayResultTmp[localId]
                                            + sharedArraySquaredQueries[sharedArraySquaredOffsetQuery + (n / 16) * 16 + tile16.meta_group_rank() + j];
                }
#elif Q == 32
                {
                    localId = sharedArrayOffset + (j + tile8.meta_group_rank()) * 8 + tile8.thread_rank();
                    sharedArrayResult[localId] = sharedArrayResult[localId]
                                                 + sharedArrayResultTmp[localId]
                                                 + sharedArraySquaredQueries[sharedArraySquaredOffsetQuery +
                                                                             (n / 16) * 32 +
                                                                             tile8.meta_group_rank() + j];
                }
#else
                {
                localId = sharedArrayOffset + (j) * 32 + warp.thread_rank();
                sharedArrayResult[localId] = sharedArrayResult[localId]
                                            + sharedArrayResultTmp[localId]
                                            + sharedArraySquaredQueries[sharedArraySquaredOffsetQuery + (n / 16) * 8 + j];
            }
#endif

            }
        } // for COMPUTE_DIM

        for (unsigned int j = 0; j < Q; j += WARP_SIZE / C) {

#if Q == 16
            {
                if ((j + tile16.meta_group_rank()) < nbQueriesBatch && tile16.thread_rank() < nbCandidatesCurrent)
                {
                ACCUM_TYPE tmpDistance = abs(sharedArrayResult[sharedArrayOffset + (j + tile16.meta_group_rank()) * 16 + tile16.thread_rank()]);
#if ACCUM_PREC == 16
                if (hsqrt(tmpDistance) <= (*epsilon))
#else
                if (sqrt(tmpDistance) <= (*epsilon))
#endif
                {
                    unsigned int tmpIdx = atomicAdd(cnt, int(1));
                }
                }
            }
#elif Q == 32
            {
                if ((j + tile8.meta_group_rank()) < nbQueriesBatch && tile8.thread_rank() < nbCandidatesCurrent) {
                    ACCUM_TYPE tmpDistance = abs(
                            sharedArrayResult[sharedArrayOffset + (j + tile8.meta_group_rank()) * 8 +
                                              tile8.thread_rank()]);
#if ACCUM_PREC == 16
                    if (hsqrt(tmpDistance) <= (*epsilon))
#else
                    if (sqrt(tmpDistance) <= (*epsilon))
#endif
                    {
                        unsigned int tmpIdx = atomicAdd(cnt, int(1));
                    }
                }
            }
#else
            {
                if ((j) < nbQueriesBatch && warp.thread_rank() < nbCandidatesCurrent)
                {
                ACCUM_TYPE tmpDistance = abs(sharedArrayResult[sharedArrayOffset + (j) * 32 + warp.thread_rank()]);
#if ACCUM_PREC == 16
                if (hsqrt(tmpDistance) <= (*epsilon))
#else
                if (sqrt(tmpDistance) <= (*epsilon))
#endif
                {
                    unsigned int tmpIdx = atomicAdd(cnt, int(1));
                }
                }
            }
#endif

        }
    } // for nbQueryPoints

}


//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\


#if COMPUTE_PREC == 64

__global__ void distanceCalculationBruteForceTensorDouble(
    unsigned int* nbQueryPoints,
    double* dataset,
    double* epsilon,
    unsigned long long* cnt,
    double* preComputedSquaredCoordinates)
{
    __shared__ double sharedArrayQueryPoints[WARP_PER_BLOCK * 8 * COMPUTE_DIM];
    // __shared__ double sharedArrayTmp8x4[WARP_PER_BLOCK * 8 * 4];
    __shared__ double sharedArraySquaredQueries[WARP_PER_BLOCK * 8 * (COMPUTE_DIM / 4)];
    __shared__ double sharedArraySquaredCandidates[WARP_PER_BLOCK * 8];
    __shared__ double sharedArrayResult[WARP_PER_BLOCK * 8 * 8];
    __shared__ double sharedArrayResultTmp[WARP_PER_BLOCK * 8 * 8];

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warpIdInGrid = tid / WARP_SIZE;
    unsigned int queryPoint = warpIdInGrid * 8;
    unsigned int warpIdInBlock = threadIdx.x / WARP_SIZE;

    unsigned int print = 1;

    if ((*nbQueryPoints) <= queryPoint)
    {
        return;
    }

    unsigned int sharedArrayQueryOffset = warpIdInBlock * 8 * COMPUTE_DIM;
    // unsigned int sharedArray8x4Offset = warpIdInBlock * 8 * 4;
    unsigned int sharedArraySquaredOffset = warpIdInBlock * 8 * (COMPUTE_DIM / 4);
    unsigned int sharedArrayOffset = warpIdInBlock * 8 * 8;

    thread_block_tile<32> warp = tiled_partition<32>(this_thread_block());
    thread_block_tile<8> tile8 = tiled_partition<8>(warp);
    thread_block_tile<4> tile4 = tiled_partition<4>(warp);
    
    unsigned int nbStepsToPage = ceil((1.0 * COMPUTE_DIM) / (1.0 * WARP_SIZE));
    unsigned int nbQueriesBatch = (queryPoint + 8 > (*nbQueryPoints)) ? (*nbQueryPoints) - queryPoint : 8;

    // Page query points
    if (tile4.meta_group_rank() < nbQueriesBatch)
    {
        for (unsigned int i = 0; i < COMPUTE_DIM; i += 4)
        {
            if ((tile4.thread_rank() + i) < COMPUTE_DIM)
            {
                sharedArrayQueryPoints[sharedArrayQueryOffset + tile4.meta_group_rank() * COMPUTE_DIM + tile4.thread_rank() + i] =
                    dataset[(queryPoint + tile4.meta_group_rank()) * COMPUTE_DIM + tile4.thread_rank() + i];
            }
        }
    } else {
        for (unsigned int i = 0; i < COMPUTE_DIM; i += 4)
        {
            if ((tile4.thread_rank() + i) < COMPUTE_DIM)
            {
                sharedArrayQueryPoints[sharedArrayQueryOffset + tile4.meta_group_rank() * COMPUTE_DIM + tile4.thread_rank() + i] = 0.0;
            }
        }
    }

    if (warp.thread_rank() < 8)
    {
        for (unsigned int i = 0; i < (COMPUTE_DIM / 4); ++i)
        {
            if (warp.thread_rank() < nbQueriesBatch)
            {
                sharedArraySquaredQueries[sharedArraySquaredOffset + i * 8 + warp.thread_rank()] =
                    preComputedSquaredCoordinates[(queryPoint + warp.thread_rank()) * (COMPUTE_DIM / 4) + i];
            } else {
                sharedArraySquaredQueries[sharedArraySquaredOffset + i * 8 + warp.thread_rank()] = 0.0;
            }
        }
    }

    for (unsigned int i = 0; i < (*nbQueryPoints); i += 8)
    {
        unsigned int nbCandidatesLeft = (*nbQueryPoints) - i;
        unsigned int nbCandidatesCurrent = min(8, nbCandidatesLeft);

        sharedArrayResult[sharedArrayOffset + warp.thread_rank()] = 0.0;
        sharedArrayResult[sharedArrayOffset + warp.thread_rank() + 32] = 0.0;

        for (unsigned int n = 0; n < COMPUTE_DIM; n += 4)
        {
            if (warp.thread_rank() < 8)
            {
                if (warp.thread_rank() < nbCandidatesCurrent)
                {
                    sharedArraySquaredCandidates[warpIdInBlock * 8 + warp.thread_rank()] =
                        preComputedSquaredCoordinates[(i + warp.thread_rank()) * (COMPUTE_DIM / 4) + (n / 4)];
                } else {
                    sharedArraySquaredCandidates[warpIdInBlock * 8 + warp.thread_rank()] = 0.0;
                }
            }

            wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> matrixQ;
            wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::col_major> matrixC;
            wmma::fragment<wmma::accumulator, 8, 8, 4, double> matrixC2;
            wmma::fragment<wmma::accumulator, 8, 8, 4, double> matrixQCC2;

            wmma::load_matrix_sync(matrixQ, sharedArrayQueryPoints + sharedArrayQueryOffset + n, COMPUTE_DIM);
            // wmma::load_matrix_sync(matrixC, sharedArrayTmp8x4 + sharedArray8x4Offset, 4);
            wmma::load_matrix_sync(matrixC, dataset + i * COMPUTE_DIM + n, COMPUTE_DIM);
            wmma::load_matrix_sync(matrixC2, sharedArraySquaredCandidates + warpIdInBlock * 8, 0, wmma::mem_row_major);
            wmma::fill_fragment(matrixQCC2, 0.0);

            for (unsigned int k = 0; k < matrixQ.num_elements; ++k)
            {
                matrixQ.x[k] *= (-2.0);
            }

            wmma::mma_sync(matrixQCC2, matrixQ, matrixC, matrixC2);
            wmma::store_matrix_sync(sharedArrayResultTmp + sharedArrayOffset, matrixQCC2, 8, wmma::mem_row_major);

            sharedArrayResult[sharedArrayOffset + tile8.meta_group_rank() * 8 + tile8.thread_rank()] =
                    sharedArrayResult[sharedArrayOffset + tile8.meta_group_rank() * 8 + tile8.thread_rank()]
                    + sharedArrayResultTmp[sharedArrayOffset + tile8.meta_group_rank() * 8 + tile8.thread_rank()]
                    + sharedArraySquaredQueries[sharedArraySquaredOffset + (n / 4) * 8 + tile8.meta_group_rank()];
            sharedArrayResult[sharedArrayOffset + tile8.meta_group_rank() * 8 + tile8.thread_rank() + 32] =
                    sharedArrayResult[sharedArrayOffset + tile8.meta_group_rank() * 8 + tile8.thread_rank() + 32]
                    + sharedArrayResultTmp[sharedArrayOffset + tile8.meta_group_rank() * 8 + tile8.thread_rank() + 32]
                    + sharedArraySquaredQueries[sharedArraySquaredOffset + (n / 4) * 8 + tile8.meta_group_rank() + 4];

            print = 0;
        } // for COMPUTE_DIM

        if (tile8.meta_group_rank() < nbQueriesBatch && tile8.thread_rank() < nbCandidatesCurrent)
        {
            double tmpDistance = fabs(sharedArrayResult[sharedArrayOffset + tile8.meta_group_rank() * 8 + tile8.thread_rank()]);
            if (sqrt(tmpDistance) <= (*epsilon))
            {
                unsigned int tmpIdx = atomicAdd(cnt, int(1));
            }
        }
        if ((tile8.meta_group_rank() + 4) < nbQueriesBatch && tile8.thread_rank() < nbCandidatesCurrent)
        {
            double tmpDistance = fabs(sharedArrayResult[sharedArrayOffset + 32 + tile8.meta_group_rank() * 8 + tile8.thread_rank()]);
            if (sqrt(tmpDistance) <= (*epsilon))
            {
                unsigned int tmpIdx = atomicAdd(cnt, int(1));
            }
        }

    } // for nbQueryPoints
}

#endif // COMPUTE_PREC == 64