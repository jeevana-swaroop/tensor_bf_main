#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <math.h>

#include "omp.h"
#include <cuda.h>

#include "params.h"
#include "main.h"
#include "dataset.h"
#include "gpu_join.h"

int main(int argc, char* argv[])
{
    char filename[256];
    strcpy(filename, argv[FILENAME_ARG]);
    ACCUM_TYPE epsilon = atof(argv[EPSILON_ARG]);
    unsigned int searchMode = atoi(argv[SEARCHMODE_ARG]);
    unsigned int device = 0;
    if (5 <= argc)
    {
        device = atoi(argv[DEVICE_ARG]);
    }

    cudaSetDevice(device);

    /***** Import dataset *****/
    std::vector< std::vector<INPUT_DATA_TYPE> > inputVector;
    double tStartReadDataset = omp_get_wtime();
    importDataset(&inputVector, filename);
    double tEndReadDataset = omp_get_wtime();
    double timeReadDataset = tEndReadDataset - tStartReadDataset;
    std::cout << "[Main | Time] ~ Time to read the dataset: " << timeReadDataset << '\n';
    unsigned int nbQueryPoints = inputVector.size();

    INPUT_DATA_TYPE* database = new INPUT_DATA_TYPE[(nbQueryPoints + ADDITIONAL_POINTS) * COMPUTE_DIM];
    for (unsigned int i = 0; i < nbQueryPoints; ++i)
    {
        for (unsigned int j = 0; j < INPUT_DATA_DIM; ++j)
        {
            database[i * COMPUTE_DIM + j] = inputVector[i][j];
        }
        for (unsigned int j = INPUT_DATA_DIM; j < COMPUTE_DIM; ++j)
        {
            database[i * COMPUTE_DIM + j] = (INPUT_DATA_TYPE) 0.0;
        }
    }
    for (unsigned int i = 0; i < ADDITIONAL_POINTS; ++i)
    {
        for (unsigned int j = 0; j < COMPUTE_DIM; ++j)
        {
            database[(nbQueryPoints + i) * COMPUTE_DIM + j] = (INPUT_DATA_TYPE) 0.0;
        }
    }

    /***** Compute the distance similarity join *****/
    double tStartJoin = omp_get_wtime();
    uint64_t totalResult = 0;
    switch (searchMode)
    {
        case SM_NVIDIA:
        {
            // We need to transpose the dataset for this searchMode
//            INPUT_DATA_TYPE* datasetTranspose = new INPUT_DATA_TYPE[nbQueryPoints * COMPUTE_DIM];
//            for (unsigned int i = 0; i < nbQueryPoints; ++i)
//            {
//                for (unsigned int j = 0; j < INPUT_DATA_DIM; ++j)
//                {
//                    datasetTranspose[j * nbQueryPoints + i] = inputVector[i][j];
//                }
//                for (unsigned int j = INPUT_DATA_DIM; j < COMPUTE_DIM; ++j)
//                {
//                    datasetTranspose[j * nbQueryPoints + i] = (INPUT_DATA_TYPE) 0.0;
//                }
//            }
//            GPUJoinMainBruteForceNvidia(searchMode, device, database, datasetTranspose, &nbQueryPoints, &epsilon, &totalResult);
//            break;
            std::cerr << "[Main] ~ Error: removed\n";
            return 1;
        }
        case SM_GPU:
        case SM_CUDA_ALT:
        case SM_TENSOR:
        case SM_TENSOR_MQ:
        {
            if(Q*C!=256 || Q%8!=0 || C%8!=0) {

                std::cout << "Set Q & C values as (16 & 16) or (32 & 8) or (8 & 32)"<<std::endl;
                return 1;
            }
            GPUJoinMainBruteForce(searchMode, device, database, &nbQueryPoints, &epsilon, &totalResult);
            break;
        }
        case SM_CPU:
        {
            std::cerr << "[Main] ~ Error: Work in Progress\n";
//            CPUJoinMainBruteForce(searchMode, database, &nbQueryPoints, &epsilon, &totalResult);
            return 1;
        }
        default:
        {
            std::cerr << "[Main] ~ Error: Unknown search mode\n";
            return 1;
        }
    }

    double tEndJoin = omp_get_wtime();
    double timeJoin = tEndJoin - tStartJoin;
    std::cout << "[Main | Result] ~ Time to join: " << timeJoin << '\n';
    std::cout << "[Main | Result] ~ Total result set size: " << totalResult << '\n';

    std::ofstream outputResultFile;
    std::ifstream inputResultFile("tensor_brute-force.txt");
    outputResultFile.open("tensor_brute-force.txt", std::ios::out | std::ios::app);
    if (inputResultFile.peek() == std::ifstream::traits_type::eof())
    {
        outputResultFile << "Dataset, epsilon, searchMode, executionTime, totalNeighbors, inputDim, computeDim, blockSize, warpPerBlock, computePrec, accumPrec\n";
    }
    outputResultFile << filename << ", " << epsilon << ", " << searchMode << ", " << timeJoin << ", " << totalResult << ", "
                    << INPUT_DATA_DIM << ", " << COMPUTE_DIM << ", " << BLOCKSIZE << ", " << WARP_PER_BLOCK << ", "
                    << COMPUTE_PREC << ", " << ACCUM_PREC << std::endl;

    outputResultFile.close();
    inputResultFile.close();
}
